#!/bin/bash
# Manual download helper — opens all insurer download pages in browser

mkdir -p data/raw_pdfs

# Open all download pages in default browser (macOS)
open "https://www.starhealth.in/downloads"
open "https://www.hdfcergo.com/health-insurance"
open "https://www.icicilombard.com/health-insurance"
open "https://www.nivabupa.com/downloads"
open "https://www.careinsurance.com/downloads"
open "https://www.adityabirlacapital.com/healthinsurance/downloads"
open "https://www.bajajgeneralinsurance.com/health-insurance-plans/health-insurance-documents.html"
open "https://www.tataaig.com/health-insurance"
open "https://www.reliancegeneral.co.in/Insurance/Health-Insurance.aspx"
open "https://www.royalsundaram.in/downloads"
open "https://www.sbigeneral.in/health-insurance-downloads"
open "https://www.manipalcigna.com/resources"
open "https://www.futuregenerali.in/health-insurance-downloads"
open "https://www.universalsompo.com/health-insurance-downloads"
open "https://www.libertyinsurance.in/health-insurance-downloads"
open "https://www.newindia.co.in/health-insurance-downloads"
open "https://www.orientalinsurance.org.in/health-insurance-downloads"
open "https://www.uiic.co.in/health-insurance-downloads"
open "https://www.nationalinsurance.nic.co.in/health-insurance-downloads"
open "https://www.godigit.com/health-insurance/policy-wordings"
open "https://www.acko.com/health-insurance/policy-wordings/"

echo "All download pages opened in browser."
echo "Manually download PDFs and save to: $(pwd)/data/raw_pdfs/"
echo "Name format: {insurer}_{policy_name}.pdf"