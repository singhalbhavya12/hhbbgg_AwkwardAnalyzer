from WMCore.Configuration import Configuration
config = Configuration()

# ------------------- General -------------------
config.section_("General")
config.General.requestName = 'analyzer_GGJets_v1'   # unique name for this submission
config.General.workArea = 'crab_projects'           # local directory for CRAB metadata

# ------------------- JobType -------------------
config.section_("JobType")
config.JobType.pluginName = 'PrivateMC'              # generic job type
config.JobType.psetName = ''                        # no CMSSW config needed
config.JobType.scriptExe = 'run_job.sh'             # your wrapper script
config.JobType.inputFiles = [
    'analyzer.py',
    '/uscms/home/bsinghal/nobackup/analysis/parquet_files/sim/bkg/merged/GGJets_MGG-80.parquet',                                   # your python script
    'run_job.sh'                                    # wrapper
]
config.JobType.outputFiles = ['*.root', '*.parquet', '*.txt']  # all output types
config.JobType.allowUndistributedCMSSW = True       # crucial for non-CMSSW jobs

# ------------------- Data -------------------
config.section_("Data")
config.Data.userInputFiles = [
    '/uscms/home/bsinghal/nobackup/analysis/parquet_files/sim/bkg/merged/GGJets_MGG-80.parquet'
]
#config.Data.inputDataset = ''                       # empty, since we are using local input
config.Data.splitting = 'FileBased'                 # one job per input file (change if multiple files)
config.Data.unitsPerJob = 1
config.Data.outputPrimaryDataset = 'AnalyzerOutput'
config.Data.outLFNDirBase = '/store/user/bsinghal/analyzer_GGJets/'  # change <USERNAME> to your LPC username
config.Data.publication = False                     # set True only if you want to publish the dataset

# ------------------- Site -------------------
config.section_("Site")
config.Site.storageSite = 'T3_US_FNALLPC'          # output destination
config.Site.whitelist = ['T3_US_FNALLPC']