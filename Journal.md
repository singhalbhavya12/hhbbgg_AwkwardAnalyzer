# Plot Features

>> python /eos/user/b/bsinghal/analysis/bbgg_low_res/hhbbgg_AwkwardAnalyzer/bs_plot_scripts/kinematics_lowmass.py
" Outputs saved in /eos/user/b/bsinghal/analysis/www/CUA/XYH/signal/kinematics/2022preEE_2022postEE/grouped_bf and individual/ "

" Run following script to combine the kinematic plots of grouped mass points and individual mass points for comparision and boost_factor based grpouping validation "
>> python /eos/user/b/bsinghal/analysis/bbgg_low_res/hhbbgg_AwkwardAnalyzer/bs_plot_scripts/make_combined_kinematics_figure.py
" Output in /eos/user/b/bsinghal/analysis/www/CUA/XYH/signal/kinematics/2022preEE_2022postEE/combined/ " 


# BDT Training
>> python /eos/user/b/bsinghal/analysis/bbgg_low_res/hhbbgg_AwkwardAnalyzer/ML_Application/low_mass/bdt/bdt_lowmass.py
" Outputs saved in /eos/user/b/bsinghal/analysis/bbgg_low_res/hhbbgg_AwkwardAnalyzer/ML_Application/low_mass/bdt/bdt_output/ "

Stick to 1000 iterations or estimators for. now
- Generated low mass and individual mass point bdt models for low mass category MC
- Try fewer features now, 