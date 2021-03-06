#!/usr/bin/env node
const fs = require('fs')
const events = require('events')
const mapYaml = require('./yaml')
const freqgen = require('@freqgen/core')
const ora = require('ora')
const Fasta = require('biojs-io-fasta')
const addMaps = require('./addMaps')
const yaml = require('js-yaml')
var program = require('commander')

function commaSeparatedIntList(value) {
  return [...new Set(value.split(',').map(x => parseInt(x)))] // deduplicate the list
}

program
  .command('featurize [files...]')
  .description('Featurize one or more FASTA files')
  .option(
    '-k, --k-mers <int>',
    'comma separated list of k values to featurize, e.g. "-k 1,2,3"',
    commaSeparatedIntList
  )
  .option('-c, --codons', 'whether to featurize codons')
  .option('-o, --output <file>', 'the output YAML file')
  .action(function(files, options) {
    // start up a pretty spinner (and correctly use file vs files!)
    const spinner = ora(
      `Parsing ${files.length} file${files.length > 1 ? 's' : ''}...`
    ).start()

    if (options.kMers == null) {
      spinner.fail(
        'No k-mers or codons specified to featurize. Provide at least one k value after -k or use the -c flag to featurize codons.'
      )
      return
    }

    // parse the FASTA files into a flat list. Ex: ["ATGC...", "GTCAA...", ...]
    let seqs = files
      .map(file =>
        Fasta.parse(fs.readFileSync(file, 'utf8')).map(obj => obj.seq)
      )
      .flat()

    let totalKmerCounts = new Map() // will map k values to maps with k-mers and counts. Ex: {1: {"A": n, "T": n}...}
    for (let k of options.kMers) {
      spinner.text = `Counting ${k}-mers for ${seqs.length} sequence(s)...`
      for (let seq of seqs) {
        let counts = freqgen.kmerCounts(freqgen.kmers(seq, k, true))

        if (totalKmerCounts.get(k)) {
          totalKmerCounts.set(k, addMaps(totalKmerCounts.get(k), counts))
        } else {
          totalKmerCounts.set(k, counts)
        }
      }
    }

    // if codon featurization is requested, count the codons of every seq
    if (!(options.codons == null)) {
      spinner.text = `Counting codons for ${seqs.length} sequence(s)...`
      for (let seq of seqs) {
        let counts = freqgen.kmerCounts(
          freqgen.kmers(seq, 3, { overlap: false })
        )

        if (totalKmerCounts.get('codons')) {
          totalKmerCounts.set(
            'codons',
            addMaps(totalKmerCounts.get('codons'), counts)
          )
        } else {
          totalKmerCounts.set('codons', counts)
        }
      }
    }

    let kmerFrequencies = new Map()

    for (let entry of totalKmerCounts.entries()) {
      kmerFrequencies.set(entry[0], freqgen.kmerFrequencies(entry[1]))
    }

    spinner.succeed(
      `Done featurizing ${files.length} file${
        files.length > 1 ? 's' : ''
      } with ${seqs.length} sequence(s)! ${
        options.output == null
          ? ''
          : 'Output written to ' + options.output + '.'
      }`
    )

    // either write to a file or print it out
    if (options.output == null) {
      console.log(yaml(kmerFrequencies))
    } else {
      fs.writeFileSync(options.output, mapYaml(kmerFrequencies))
    }
  })
program
  .command('generate')
  .description(
    'Given a set of target k-mer and/or codon frequencies and an amino acid sequence, generate a DNA sequence.'
  )
  .option(
    '-s, --seq <file>',
    'the input FASTA file containing the amino acid sequence'
  )
  .option(
    '-f, --freq <file>',
    'the input YAML file containing the k-mer and/or codon frequencies to target'
  )
  .option(
    '-o, --output <file>',
    'the output FASTA file (if not provided, writes to stdout)'
  )
  .option(
    '-g, --genetic-code <int>',
    'the genetic code to use (default: 11)',
    11
  )
  .option(
    '-p, --pop-size <int>',
    'the size of the population (default: 100)',
    100
  )
  .option('-m, --mutation-rate <int>', 'the mutation rate (default: 0.3)', 0.3)
  .option(
    '-c, --crossover-rate <int>',
    'the crossover rate for the population (default: 0.8)',
    0.8
  )
  .option(
    '-e, --early-stopping <int>',
    'after how many generations without at least --rel-tol percent improvement to stop the optimization (default: 50)',
    50
  )
  .option(
    '-r, --rel-tol <float>',
    'the percentage increase required to reset the early stopping counter (default: 0.0001, must be in [0, 1])',
    0.0001
  )
  .option(
    '--pop-count <int>',
    'how many populations to optimize, returning the best result (default: 1)',
    1
  )
  .option('--no-cache', 'whether to disable result caching (default: false)')
  .option(
    '--log [file]',
    'whether to log JSON metadata to a file (default: false)'
  )
  .option(
    '--no-metadata',
    "whether to exclude JSON metadata from the output FASTA's comment line (default: false)"
  )
  .option(
    '--dna',
    'whether to interpret the FASTA file as DNA (default: false)'
  )
  .action(function(options) {
    // read in the freqs
    const spinner = ora(
      `Reading target frequencies from ${options.freq}`
    ).start()
    freqs = Object.entries(yaml.safeLoad(fs.readFileSync(options.freq, 'utf8')))
    freqs = freqs.map(x => [
      x[0] == 'codons' ? 'codons' : Number(x[0]),
      new Map(Object.entries(x[1])),
    ])
    freqs = new Map(freqs)

    // read in the seq
    spinner.text = `Reading target amino acid sequence from ${options.seq}`
    seq = Fasta.parse(fs.readFileSync(options.seq, 'utf8'))[0].seq

    // translate the sequence, if needed
    if (options.dna) {
      seq = freqgen.translate(seq, options.geneticCode)
    }

    // show progress updates
    var emitter = new events.EventEmitter()
    emitter.on('generation', x => {
      // don't show the details when optimizing multiple populations
      if (options.popCount > 1) {
        return
      }
      spinner.text = `Generation number:\t${
        x.iterationNumber
      }\n  Current fitness:\t${x.bestIndividualFitness.toFixed(
        4
      )}\n  Since increase:\t${x.gensSinceImprovement}`
    })

    const evolvePopulations = async () => {
      // keep track of each population's best individual
      fittestInPopulations = []
      start = new Date()
      for (let index = 0; index < options.popCount; index++) {
        // update the spinner
        spinner.text = `Optimizing population ${index + 1}/${options.popCount}`
        // GenAlgo is async, so we have to wait
        result = await freqgen.generate(seq, freqs, {
          cache: options.cache,
          emitter,
          mutationProbability: Number(options.mutationRate),
          relTol: Number(options.relTol),
          crossoverProbability: Number(options.crossoverRate),
          populationSize: Number(options.popSize),
          maxGensSinceImprovement: Number(options.earlyStopping),
        })
        // store the result
        fittestInPopulations.push(result[0])
      }
      stop = new Date() - start

      let metadata = {
        fitness: Number(fittestInPopulations[0].fitness.toFixed(4)),
        unixTimestamp: Date.now(),
        durationMilliseconds: stop,
        sequenceFile: options.seq,
        target: options.freq,
        mutationRate: options.mutationRate,
        crossoverRate: options.crossoverRate,
        populationSize: options.popSize,
        populationCount: options.popCount,
        earlyStopping: options.earlyStopping,
        relTol: options.relTol,
      }
      if (options.log) {
        try {
          fs.writeFileSync(options.log, JSON.stringify(metadata))
        } catch (error) {
          fs.writeFileSync(
            `${
              options.output ? options.output : 'freqgen-' + Date.now()
            }.log.json`,
            JSON.stringify(metadata)
          )
        }
      }
      spinner.succeed(
        `Done! Generated a DNA sequence with fitness ${fittestInPopulations[0].fitness.toFixed(
          4
        )} in ${(stop / 1000).toFixed(2)}s.`
      )
      let outputFasta = `>Generated by Freqgen from ${options.seq} targeting ${options.freq}.`
      if (options.metadata) {
        outputFasta += ` Metadata: ${JSON.stringify(metadata)}.`
      }
      outputFasta += `\n${fittestInPopulations[0].entity}`

      if (options.output) {
        fs.writeFileSync(options.output, outputFasta)
      } else {
        console.log(outputFasta)
      }
      return fittestInPopulations[0]
    }

    evolvePopulations().catch(e => spinner.fail(e.message))
  })
program.parse(process.argv)
