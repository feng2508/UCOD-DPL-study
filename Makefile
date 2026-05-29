ZIP_NAME := UCOD-DPL-kaggle.zip
DATASETS_ZIP_NAME := datasets.zip

.PHONY: zip-kaggle zip-datasets

zip-kaggle:
	rm -f $(ZIP_NAME)
	zip -r $(ZIP_NAME) . \
		-x "datasets/*" \
		-x "$(ZIP_NAME)" \
		-x "$(DATASETS_ZIP_NAME)" \
		-x ".git/*" \
		-x "__pycache__/*" \
		-x "*/__pycache__/*" \
		-x ".DS_Store" \
		-x "*/.DS_Store" \
		-x "work_dir/*" \
		-x "results/*"

zip-datasets:
	rm -f $(DATASETS_ZIP_NAME)
	zip -r $(DATASETS_ZIP_NAME) datasets \
		-x ".DS_Store" \
		-x "*/.DS_Store"
