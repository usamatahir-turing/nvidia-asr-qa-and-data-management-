GCS Instance: [https://console.cloud.google.com/agent-platform/workbench/locations/us-central1-a/instances/nvidia-asr-qa-and-data-management?project=delivery-nvidia](https://console.cloud.google.com/agent-platform/workbench/locations/us-central1-a/instances/nvidia-asr-qa-and-data-management?project=delivery-nvidia)

For any access issues, here’s the thread with DevOps: [Slack Thread](https://turing-company.slack.com/archives/C02TKAK7Q83/p1779742376834579?thread_ts=1778572701.736009&cid=C02TKAK7Q83)

# Mirror Drive {#mirror-drive}

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

# Segment Annotation Quality

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

For uploading reports ran on the approved Seglists:

```shell
python push_reports_to_drive.py --variant approved
```

The data will be pushed to the relevant subfolder in the drive [Files for Gecko](https://drive.google.com/drive/folders/1D8isShidIb1hcZuCezV-Qe7EsmsmKBR1)

*Note: It’s important that for any uploading of data, the folder is in a Shared Drive location (not shared with me)*

