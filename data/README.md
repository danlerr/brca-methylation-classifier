# Data

The dataset is **not included** in this repository.

The code expects a single CSV file at `data/raw/total_df.csv` (path configurable in
`configs/default.yaml`) with one row per patient and the following columns:

| Column(s) | Description |
|---|---|
| `ID_paziente` | patient identifier |
| `<chr>:<arm>:i<k>` (e.g. `17:q:i3`) | methylation features, 5 intervals per chromosome arm (220 columns) |
| `<chr><arm>_{del,shallow,gain,ampl}` | copy-number aberrations per chromosome arm |
| `<chr><arm>_{SNP,DEL,INS,ONP}` | mutation counts per chromosome arm |
| `<chr>:<arm>_tot` | per-arm totals |
| `acronyms_y` | tumour type (TCGA acronym); `brca` is the positive class |
| `acronyms_x`, `extended_x`, `extended_y` | redundant labels, dropped |

8,256 patients, 648 features, ~8.4% BRCA.
