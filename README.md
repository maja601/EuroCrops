![03_bratislava4_sw_border_parcels](https://user-images.githubusercontent.com/22978370/161757196-c0316b58-6ee8-48a3-a604-4d9aafc3adb4.png)
_Border Region Austria - Slovakia around Bratislava_
# EuroCrops
[![CC BY 4.0][cc-by-shield]][cc-by]

EuroCrops is a dataset collection combining all publicly available self-declared crop reporting datasets from countries of the European Union.
The project is funded by the German Space Agency at DLR on behalf of the Federal Ministry for Economic Affairs and Climate Action (BMWK).
This work is licensed under a [Creative Commons Attribution 4.0 International License][cc-by].

Right now EuroCrops only includes vector data, but stay tuned for a version that includes satellite imagery!

***
## Content

1. [Background](#background)
2. [Participating countries](#participating_countries)
3. [GitHub project structure](#github_structure)
4. [Vector data download](#vectordata)
5. [Reference](#reference)


***
## Background <a name="background"></a>
![g1924](https://user-images.githubusercontent.com/22978370/158154235-06794acb-d163-447e-80ca-16e9cd60f11c.png)

**Disclaimer**: The *Nomenclature of Territorial Units for Statistics 3 (NUTS3)* region, which we added by hand, is just an approximate assignment of a crop parcel to a region.
It might happen that a parcel is not correctly allocated to the right region or country.
The NUTS3 attribute is only meant to be an aid for a meaningful spatial division of the dataset into training, validation and test sets.

## Participating countries <a name="participating_countries"></a>
<p align="center"><img width=43.5% src="https://user-images.githubusercontent.com/22978370/157669864-2d5d0df7-1fb0-40b6-ace3-a625cfef6195.png"></p>

Find detailed information for all countries of the European Union in our [Wiki](https://github.com/maja601/EuroCrops/wiki), especially the countries represented in EuroCrops:
- [Austria](https://github.com/maja601/EuroCrops/wiki/Austria)
- [Belgium](https://github.com/maja601/EuroCrops/wiki/Belgium)
- [Germany](https://github.com/maja601/EuroCrops/wiki/Germany)
- [Denmark](https://github.com/maja601/EuroCrops/wiki/Denmark)
- [Estonia](https://github.com/maja601/EuroCrops/wiki/Estonia)
- [Spain](https://github.com/maja601/EuroCrops/wiki/Spain)
- [France](https://github.com/maja601/EuroCrops/wiki/France)
- [Croatia](https://github.com/maja601/EuroCrops/wiki/Croatia)
- [Lithuania](https://github.com/maja601/EuroCrops/wiki/Lithuania)
- [Latvia](https://github.com/maja601/EuroCrops/wiki/Latvia)
- [Netherlands](https://github.com/maja601/EuroCrops/wiki/Netherlands)
- [Portugal](https://github.com/maja601/EuroCrops/wiki/Portugal)
- [Romania](https://github.com/maja601/EuroCrops/wiki/Romania)
- [Sweden](https://github.com/maja601/EuroCrops/wiki/Sweden)
- [Slovenia](https://github.com/maja601/EuroCrops/wiki/Slovenia)
- [Slovakia](https://github.com/maja601/EuroCrops/wiki/Slovakia)

## GitHub project structure <a name="github_structure"></a>
```
├── csvs
│   ├── country_mappings
│       └── [CSV mapping files for all participating countries]
└── hcat_core
    └── HCAT.csv
```

## Vector data download<a name="vectordata"></a>
The shapefiles of the countries are available via [Sync&Share](https://syncandshare.lrz.de/getlink/fiAD95cTrXbnKMrdZYrFFcN8/). They might be still some changes with the data in this folder, such as adding the NUTS3 region identifier to the parchels. Version-controlled and citable editions will be made available in near future over [mediaTUM](https://mediatum.ub.tum.de/).
```
├── AT
│   └── AT_2021_EC21.*
├── BE
│   └── VLG
│       └── BE_VLG_2021_EC21.*
├── DE
│   ├── LS
│   |   └── DE_LS_2021_EC21.*
│   └── NRW
│       └── DE_NRW_2021_EC21.*
├── DK
│   └── DK_2019_EC21.*
├── EE
│   └── EE_2021_EC21.*
├── ES
│   └── NA
│       └── ES_NA_2020_EC21.*
├── FR
│   └── FR_2018_EC21.*
├── HR
│   └── HR_2020_EC21.*
├── LT
│   └── LT_2021_EC.*
├── LV
│   └── LV_2021_EC21.*
├── NL
│   └── NL_2021_EC21.*
├── PT
│   └── PT_2021_EC21.*
├── RO
│   └── RO_ny_EC21.*
├── SE
│   └── SE_2021_EC21.*
├── SI
│   └── SI_2021_EC21.*
└── SK
    └── SK_2021_EC21.*
```

## Reference<a name="reference"></a>


[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
