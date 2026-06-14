GCS Instance: [https://console.cloud.google.com/agent-platform/workbench/locations/us-central1-a/instances/nvidia-asr-qa-and-data-management?project=delivery-nvidia](https://console.cloud.google.com/agent-platform/workbench/locations/us-central1-a/instances/nvidia-asr-qa-and-data-management?project=delivery-nvidia)

For any access issues, here’s the thread with DevOps: [Slack Thread](https://turing-company.slack.com/archives/C02TKAK7Q83/p1779742376834579?thread_ts=1778572701.736009&cid=C02TKAK7Q83)

# Mirror Drive

First step should be to mirror the drive folder [Files for Gecko](https://drive.google.com/drive/u/0/folders/1D8isShidIb1hcZuCezV-Qe7EsmsmKBR1)

### Usage

Run the following in the root

```shell
python download_and_update_data.py
```

* This will mirror the Files for Gecko Folder exactly as it looks like in Drive to the folder drive\_data.  
* All changes (e.g., any new file added, deleted, or renamed) will be updated.

# Preparing Seglists and RTTMs for Delivery

By default, this process runs on the *seglst.json* files with the *\*\_approved* suffix found within the task folders of the *drive\_data* directory (which mirrors the [Files for Gecko](https://drive.google.com/drive/u/0/folders/1D8isShidIb1hcZuCezV-Qe7EsmsmKBR1) Drive folder). So it’s important that before running this script, the data in the drive\_data folder is updated as described [here](#mirror-drive)

1. ## Fixing tokens in Seglists

We know that gecko messes up the tokens (by removing square brackets or adding redundant spaces). The first step would be to fix these tokens.

### Usage

The process retrieves data from *drive\_data*, corrects the tokens in the *\*\_approved.seglst.json* files, and saves the updated data to the task folders within *seglst\_fixes\_and\_rttm\_generation/output\_data*.

```shell
cd seglst_fixes_and_rttm_generation/
python fix_seglst_tokens.py
```

2. ## Generating RTTMs using Seglists

Once the tokens have been fixed, the next step is to generate the RTTMs

### Usage

```shell
cd seglst_fixes_and_rttm_generation/
python seglst_to_rttm.py
```

This will generate .rttm files in the task folders within *seglst\_fixes\_and\_rttm\_generation/output\_data* 

3. ## Renaming to remove \_approved suffix

Finally, run this script to remove the \_approved suffix so that the .seglst.json and .rttm files are delivery ready. 

### Usage

```shell
cd seglst_fixes_and_rttm_generation/
python strip_approved_suffix.py
```

This will rename the files by removing the \_approved suffix. The final delivery-ready SegList and RTTM files will now be available in the task folders within *seglst\_fixes\_and\_rttm\_generation/output\_data* 

4. ## Pushing to GDrive for Delivery

Once all of the above is done, you can run the following script to upload everything to a GDrive folder [pre\_delivery\_folder](https://drive.google.com/drive/folders/1_tNysDjOd7MLThHQDlZeuzkrR9EgXxJf?usp=drive_link) that can be used to move the data to the final delivery folder. 

### Usage 

```shell
cd seglst_fixes_and_rttm_generation/
python upload_to_drive_for_delivery.py
```

This will upload all the RTTMs, SegLists, from *seglst\_fixes\_and\_rttm\_generation/output\_data* and their corresponding channel .wav files from tasks in *drive\_data* to a Google drive folder [pre\_delivery\_folder](https://drive.google.com/drive/folders/1_tNysDjOd7MLThHQDlZeuzkrR9EgXxJf)  
*Note: It’s important that for any uploading of data, the folder is in a Shared Drive location (not shared with me)*

To upload specific conversations only:

```shell
python upload_to_drive_for_delivery.py NV-AR-SS03-CONVO07 NV-KO-SS03-CONVO07
```

# Seglist JSON Statistics

This script computes segment-length and speech-overlap statistics from delivery-ready seglst or RTTM files in *seglst\_fixes\_and\_rttm\_generation/output\_data*. Results are written to *seglst\_fixes\_and\_rttm\_generation/turing\_json\_stats.txt*.

### Usage

```shell
cd seglst_fixes_and_rttm_generation/
python turing_json_stats.py
```

To include per-conversation breakdowns in the report:

```shell
python turing_json_stats.py --per-conversation
```

To run on specific conversations only:

```shell
python turing_json_stats.py NV-AR-SS03-CONVO07 NV-KO-SS03-CONVO07
```

By default it prefers seglst files and falls back to RTTM per speaker (`--source auto`). Use `--source seglst` or `--source rttm` to force one format.

# Final Speaker Changes

After data is in the [pre\_delivery\_folder](https://drive.google.com/drive/folders/1_tNysDjOd7MLThHQDlZeuzkrR9EgXxJf), this script copies conversations to the final delivery Drive folder. Speaker email filenames are renamed to `SPK01`, `SPK02`, etc. (alphabetical by email). It also updates speaker fields inside seglst and RTTM files. `*_mixed.wav` files are copied unchanged.

The script writes *final\_speaker\_changes/mappings.csv* before copying to Drive.

### Usage

```shell
cd final_speaker_changes/
python copy_speakers_to_final_drive.py
```

To process specific conversations only:

```shell
python copy_speakers_to_final_drive.py NV-AR-SS03-CONVO09 NV-KO-SS03-CONVO07
```

To validate without uploading to the destination Drive folder:

```shell
python copy_speakers_to_final_drive.py --dry-run NV-GR-SS04-CONVO10
```

*Note: The destination folder must be on a Shared Drive and accessible to the delivery service account (same requirement as pre-delivery upload).*

# Segment Annotation Quality

The following checks are currently in place and should be present in each generated report:
- **Boundary failures** — a segment’s start or end is off from where the speaker actually starts/stops talking by more than **100 ms**.
- **Silence failures** — there is a stretch of silence longer than **200 ms** *inside* a segment. The segment should be split there.
- **Uncovered audio** — audio is present on this speaker’s channel, but no segment is annotated. A new segment should be added.
- **No signal** — a segment exists in the annotations, but no audible audio was found in that time range on this channel. The segment may be on the wrong channel, at the wrong time, or shouldn’t exist.

## Report Generation

We want to run the segmentation quality reports at two stages

1. On the *\*\_fixed* files: these are the files that have undergone moderator’s review  
2. On the \*\_approved files: These are the files that have undergone both the moderator and the adjudicator review

### Usage

```shell
cd segment_quality_report_gen/
```

To generate the reports on \*\_fixed files and have these files ready for adjudicators, you can run the following:

```shell
python generate_report.py --variant fixed 
```

This will generate the report markdown files in the folder *segment\_quality\_report\_gen/reports\_fixed* 

To run it on the final SegList json (\*\_approved.seglst.json) files:

```shell
python generate_report.py --variant approved
```

This will generate the report markdown files in the folder *segment\_quality\_report\_gen/reports\_approved*

The names of the report md files will have the suffix of \_fixed or \_approved depending on the specified variant type

## Pushing Reports to Drive

Once reports are generated and are available in the folder *segment\_quality\_report\_gen/reports\_fixed* or *segment\_quality\_report\_gen/reports\_approved*, we can go ahead and upload them to the relevant drive folders that moderators/adjudicators can review.

### Usage

For uploading reports ran on the fixed SegLists:

```shell
cd segment_quality_report_gen/
python push_reports_to_drive.py --variant fixed
```

To upload reports for specific conversations only:

```shell
python push_reports_to_drive.py --variant fixed NV-AR-SS03-CONVO07
python push_reports_to_drive.py --variant fixed NV-AR-SS03-CONVO07 NV-KO-SS03-CONVO07
```

For uploading reports ran on the approved Seglists:

```shell
python push_reports_to_drive.py --variant approved
```

To upload approved reports for specific conversations only:

```shell
python push_reports_to_drive.py --variant approved NV-AR-SS03-CONVO07 NV-KO-SS03-CONVO07
```

* When conversation id(s) are provided, only `reports_<variant>/<CONVERSATION>_<variant>.md` files are uploaded to the matching Drive subfolder.
* When omitted, all `*_<variant>.md` files in `reports_fixed/` or `reports_approved/` are uploaded.

The data will be pushed to the relevant subfolder in the drive [Files for Gecko](https://drive.google.com/drive/folders/1D8isShidIb1hcZuCezV-Qe7EsmsmKBR1)

*Note: It’s important that for any uploading of data, the folder is in a Shared Drive location (not shared with me)*

# Token Segment Density Check

This script fixes bracket tokens in seglst files from *drive\_data*, then writes a CSV with per-segment counts of annotation tokens (e.g. `[inhale]`) versus regular words. Run it after [mirroring Drive](#mirror-drive) so *drive\_data* is up to date.

### Usage

```shell
cd token_segment_density_check/
python check_token_seg_density.py
```

Optional paths:

```shell
python check_token_seg_density.py --input ../drive_data
python check_token_seg_density.py --fixed-output ./fixed_tokens_output_folder --csv-output ./token_segment_density.csv
```

* **Phase 1** — reads `*_fixed.seglst.json` and `*_approved.seglst.json` from each task folder under `--input` (default: *drive\_data*), applies the same token normalization as `fix_seglst_tokens.py`, and writes fixed copies to *token\_segment\_density\_check/fixed\_tokens\_output\_folder*.
* **Phase 2** — scans the fixed output folder. For each speaker, prefers `*_approved.seglst.json` over `*_fixed.seglst.json` when both exist, then writes *token\_segment\_density\_check/token\_segment\_density.csv*.
* **CSV columns** — `folder_name`, `file_name`, `session_id`, `speaker`, `words`, `start_time`, `end_time`, `duration`, `segment_index`, `number_of_token_words`, `number_of_non_token_words`.
* Token words are text inside `[...]` spans; non-token words are the remaining words split on whitespace.

# AssemblyAI or LT JSONs to SegLists

This workflow is used to convert transcripts from AssemblyAI or the Labeling Tool (LT) into standardized SegList JSON files.

### 1. Download AssemblyAI JSONs
If you are working with AssemblyAI transcripts, first sync the latest JSON files from the Drive folder [AssemblyAI Transcripts](https://drive.google.com/drive/u/0/folders/1wceeL4NRLTXg57EIgV5peQPuDCBEthzl):

```shell
python assemblyai_or_LT_jsons_to_seglsts/download_jsons_from_drive.py
```
* This replicates only the immediate parent folder of each `.json` file into `assemblyai_or_LT_jsons_to_seglsts/assembly_ai_jsons`.
* It uses "updates only" logic to download new/changed files and clean up orphans.

### 2. Prepare LT JSONs
If the data is from the Labeling Tool, ensure the relevant JSON files are placed in the directory:
`assemblyai_or_LT_jsons_to_seglsts/lt_jsons/`

### 3. Generate SegLists
Once the source JSONs are ready (either in `assembly_ai_jsons/` or `lt_jsons/`), run the generator script:

```shell
cd assemblyai_or_LT_jsons_to_seglsts/
python seglst_gen.py
```
* This script will process the source JSONs and generate standardized SegList files in the `assemblyai_or_LT_jsons_to_seglsts/output_seglsts/` directory.
* One IMPORTANT caveat is that currently the Assembly AI JSONs residing in the folder have no speaker mapping. The speakers are named as A, B, C... etc. so it's important that this mapping is added first.

### 4. Upload SegLists to Files for Gecko

Upload generated seglst files from *output\_seglsts* into the [Files for Gecko](https://drive.google.com/drive/folders/1D8isShidIb1hcZuCezV-Qe7EsmsmKBR1) Drive folder. Conversation name(s) are **required**.

```shell
cd assemblyai_or_LT_jsons_to_seglsts/
python upload_seglst_json_to_gecko_folder.py NV-AR-SS03-CONVO07
python upload_seglst_json_to_gecko_folder.py NV-AR-SS03-CONVO07 NV-KO-SS03-CONVO07
```

* Uploads only `*.seglst.json` files from each local conversation folder.
* If the conversation folder already exists on Drive, it prints a warning and skips that conversation (no overwrite).
* Creates a new conversation folder on Drive when one does not exist yet.

### 5. Copy channel WAVs to Files for Gecko

After seglists are on Gecko, copy matching channel `.wav` files from the [AssemblyAI Transcripts](https://drive.google.com/drive/u/0/folders/1wceeL4NRLTXg57EIgV5peQPuDCBEthzl) Drive folder into the same conversation folders on [Files for Gecko](https://drive.google.com/drive/folders/1D8isShidIb1hcZuCezV-Qe7EsmsmKBR1). Conversation name(s) are **required**.

```shell
cd assemblyai_or_LT_jsons_to_seglsts/
python copy_channel_wavs_to_gecko_folder.py NV-AR-SS03-CONVO07
python copy_channel_wavs_to_gecko_folder.py NV-AR-SS03-CONVO07 NV-KO-SS03-CONVO07
```

* Reads `*.seglst.json` on Gecko for each conversation and derives the speaker stem (e.g. `mohamed.h2@turing.com` from `mohamed.h2@turing.com.seglst.json`).
* Finds `<speaker>.wav` in the AssemblyAI source folder on Drive (searches nested language/batch folders recursively, same layout as `download_jsons_from_drive.py`).
* Copies each WAV into the Gecko conversation folder; replaces an existing WAV with the same name.
* Warns and skips a speaker when the source WAV is missing on AssemblyAI Drive.