def get_mask_preselection(cms_events):
    mask_preselection = (cms_events.dibjet_mass > 0) & (cms_events.diphoton_mass > 0)
    return mask_preselection

#------------------------

def get_mask_selection(cms_events):
    mask_selection = (
            (cms_events.lead_isScEtaEB == 1)
            & (cms_events.sublead_isScEtaEB == 1)
        )
    return mask_selection


#-----------------
# Check https://btv-wiki.docs.cern.ch/ScaleFactors/Run3Summer22/ for the tagger point score


def get_mask_srbbgg(cms_events):    # Pass medium Btag and pass tight photonID
    mask_srbbgg = (
        (cms_events.lead_pho_mvaID_WP80 == 1)  # tight cut mvaID80
        & (cms_events.sublead_pho_mvaID_WP80 == 1)
        & (cms_events.lead_bjet_PNetB > 0.2605)
        & (cms_events.sublead_bjet_PNetB > 0.2605)
        & (cms_events.lead_isScEtaEB == 1)
        & (cms_events.sublead_isScEtaEB == 1)
    )
    return mask_srbbgg

