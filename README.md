# space-radio-data
Digital archive and summary figures for low frequency radio from space and ground. Includes RAE1,RAE2, IMP-6, Elektron, and assorted ground-based measurements from 60s to today in the band 500kHz-400MHz.

Here is an example output.

![SRD][spectrum_example.png]
The road to opening the 21cm window on the dark ages and cosmic dawn for physics and cosmology. Existing data from orbit
and ground overplotted on global 21cm signal predictions. A smooth 5th order polynomial is subtracted to illustrate
progress towards detection of spectral lines. Models of the 21cm signal show a sample of range of theoretical
possibilities.


#  Note on project methods.
This project was completed using the Claude LLM Sonnet 4.6 running in "High", the second most intense mode. This is my 
first experiment at using an LLM for a research project. It seems a reasonable use case: a fairly well defined problem requiring tedious data aggregation
and complex plotting. And I am in a hurry! This project start to finish took about 22 hours cumulative, three-ish work
days.  Unaided it would have taken easily a week or more and probably generated much more convoluted code as technical
debt built up in iteration.


While I have made all the usual
checks on the input data, fitting, and have read and edited much of the code I have not read each line. There may be
a new type of problem which we are not yet accustomed to looking for. I am pleased with the overall result and recommend
treating the result to the normal level of skepticism due to an unrefereed analysis.  

Please report issues to me Danny Jacobs (dcjacob2@asu.edu)

~Danny



All data, models, and foreground fitting methods used in this repository, with links to the NASA Astrophysics Data System (ADS).
 
---
 
## Observational Data
Much of the original space data and ground-based data were extracted from Cane 1979 to whom we owe a great debt for
gathering these numbers together. This
paper includes several data sources which are not available in modern archives including an AAS talk from 1969 and one
reference to data from a Soviet mission.

 
### Space-based
 
**Brown (1973)** — IMP-6 minimum spectrum, 0.13–2.6 MHz (`brown1973_table1.csv`)
> Brown, L. W. 1973, *ApJ*, **180**, 359–370.
> "The Galactic Radio Spectrum Between 130 and 2600 kHz"
> [ADS: 1973ApJ...180..359B](https://ui.adsabs.harvard.edu/abs/1973ApJ...180..359B)
 
**Novaco & Brown (1978)** — RAE-2, NGP/SGP/GC/GAC, 0.25–9.2 MHz (via Cane 1979, observer 15; also `NovacoBrown1978-Fig1.csv`, `NovacoBrown1978-Fig2.csv`)
> Novaco, J. C. & Brown, L. W. 1978, *ApJ*, **221**, 114–123.
> "Nonthermal Galactic Emission Below 10 Megahertz"
> [ADS: 1978ApJ...221..114N](https://ui.adsabs.harvard.edu/abs/1978ApJ...221..114N/abstract)
 
**Alexander et al. (1969a)** — RAE-1, published paper (Cane 1979 observer 3)
> Alexander, J. K., Brown, L. W., Clark, T. A., Stone, R. G., & Weber, R. R. 1969, *ApJ (Letters)*, **157**, L163.
> "The Spectrum of the Cosmic Radio Background Between 0.4 and 6.5 MHz"
> [ADS: 1969ApJ...157L.163A](https://ui.adsabs.harvard.edu/abs/1969ApJ...157L.163A)
 
**Alexander et al. (1969b)** — RAE-1, AAS 1969 meeting presentation (Cane 1979 observer 16)
> Alexander, J. K., Brown, L. W., Clark, T. A., Stone, R. G., & Weber, R. R. 1969, AAS meeting presentation, New York.
> *No ADS record; cited as unpublished in Cane (1979) Table 4.*
 
**Benediktov et al. (1965)** — Soviet satellite, 0.45–1.6 MHz (Cane 1979 observer 5)
> Benediktov, E. A., et al. 1965, *Space Research*, **5**.
> *ADS record not available; cited in Cane (1979) Table 4. Likely Elektron-series spacecraft.*
 
### Ground-based
 
**Cane (1979)** — Compilation of ground-based NGP/SGP measurements, 0.25–178 MHz (`cane1979_table2_3.csv`)
> Cane, H. V. 1979, *MNRAS*, **189**, 465–478.
> "Spectra of the Non-thermal Radio Radiation from the Galactic Polar Regions"
> [ADS: 1979MNRAS.189..465C](https://ui.adsabs.harvard.edu/abs/1979MNRAS.189..465C/abstract)
>
> *Note: Cane (1979) Tables 2 and 3 are a compilation of measurements from 24 distinct observer groups (see Cane Table 4 for the full list). The primary independent data sources represented in our datasets are Alexander et al. (1969a,b), Novaco & Brown (1978), and Benediktov et al. (1965) as space-based entries; and observers including Cottony & Johler (1952), Andrew (1966), Getmantsev et al. (1969), Purton (1966), Bridle (1967), Yates & Wielebinski (1966), and others as ground-based entries.*
 
**Bowman et al. (2018)** — EDGES high-band sky spectrum and residuals, 51–99 MHz (`edges_nature2018_figure1_plotdata.csv`)
> Bowman, J. D., Rogers, A. E. E., Monsalve, R. A., Mozdzen, T. J., & Mahesh, N. 2018, *Nature*, **555**, 67–70.
> "An Absorption Profile Centred at 78 Megahertz in the Sky-averaged Spectrum"
> [ADS: 2018Natur.555...67B](https://ui.adsabs.harvard.edu/abs/2018Natur.555...67B)
>
> Data publicly available at: https://loco.lab.asu.edu/edges/edges-data-release/
 
---
 
## Foreground Model
 
**Mozdzen et al. (2019)** — EDGES low-band foreground model, eq. 5 (3-parameter fit with ionospheric correction, LST = 6h), 50–100 MHz
> Mozdzen, T. J., Mahesh, N., Monsalve, R. A., Rogers, A. E. E., & Bowman, J. D. 2019, *MNRAS*, **483**, 4416–4428.
> "Spectral Index of the Diffuse Radio Background Between 50 and 100 MHz"
> [ADS: 2019MNRAS.483.4411M](https://ui.adsabs.harvard.edu/abs/2019MNRAS.483.4411M)
 
---
 
## 21 cm Signal Models
 
**Furlanetto, Oh & Briggs (2006)** — Standard ΛCDM dark ages 21 cm signal (`FOH2006_Tb_datasets.csv`)
> Furlanetto, S. R., Oh, S. P., & Briggs, F. H. 2006, *Phys. Rep.*, **433**, 181–301.
> "Cosmology at Low Frequencies: The 21 cm Transition and the High-Redshift Universe"
> [ADS: 2006PhR...433..181F](https://ui.adsabs.harvard.edu/abs/2006PhR...433..181F)
 
**Mondal, Barkana & Fialkov (2023)** — Dark ages 21 cm signal for standard ΛCDM and excess radio background models Ar = 0.001 and Ar = 0.4 (`Mondal2023_Fig3_Tb_datasets.csv`)
> Mondal, R., Barkana, R., & Fialkov, A. 2024, *MNRAS*, **527**, 1461–1471.
> "Constraining Exotic Dark Matter Models with the Dark Ages 21-cm Signal"
> [ADS: 2024MNRAS.527.1461M](https://ui.adsabs.harvard.edu/abs/2024MNRAS.527.1461M/abstract)
 
---
 
## Notes on Data Provenance
 
- The Cane (1979) compilation predates standardised data formats by several decades. Some entries are measurements "corrected to the galactic pole" (flagged as NGP or SGP in the declination column), meaning an attempt was made to estimate the true polar brightness temperature rather than reporting the raw antenna temperature at the pointing direction.
- The Brown (1973) IMP-6 spectrum represents the **minimum** galactic radiation observed in a broad area about the ecliptic plane. An extragalactic background model (Clark, Brown & Alexander 1970) was subtracted; this contribution is less than 1% below 1 MHz.
- The Bowman et al. (2018) `Tres1` column is the residual after subtracting a 5-term physically motivated foreground model from the sky-averaged spectrum. It does **not** include the best-fit 21 cm signal; `Tres2` contains the residual after both foreground and 21 cm signal removal.
- The Mondal et al. (2023) `Standard` model column uses an opposite sign convention from Furlanetto et al. (2006); it has been sign-flipped in `signals.toml` to place absorption signals below zero on the residual plot.


