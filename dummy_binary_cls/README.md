this script load data from huggingface dataset directly, instead of loading downloaded csv data

additional dataset config in finetune_config.json
```
    "dataset": {
        "name": "sg247/binary-classification",
        "use_test_for_validation": true, # if there is no validation split, set this to true
        "text_column" : "tweet",
        "label_column" : "label"
    }
```
