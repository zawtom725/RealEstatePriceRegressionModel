.PHONY: clean
DATA = ./data
JOB_ID="train_regression_${USER}_$(shell date +%Y%m%d_%H%M%S)"
JOB_DIR = gs://property_nn
download:
	./download.sh

unzip:
	for z in $(shell ls $(DATA)); do \
		unzip $(DATA)/$$z -d $(DATA); \
		rm -r $(DATA)/$$z; \
	done

spark: # submit pyspark job to cluster
	gsutil cp Preprocess_data.py gs://dataproc-35c19246-5eb7-4b6a-8e2f-186d38261366-sa-east1/
	gcloud dataproc jobs submit pyspark \
	gs://dataproc-35c19246-5eb7-4b6a-8e2f-186d38261366-sa-east1/Preprocess_data.py \
	--cluster data-preprocessing-cluster

preprocessed_dataset: train test
	rm train/*
	rm test/*
	gsutil cp -r gs://property_nn/train/* ./train
	gsutil cp -r gs://property_nn/test/* ./test

ml:
	gcloud ml-engine jobs submit training $(JOB_ID) \
	--job-dir $(JOB_DIR) \
	--scale-tier basic-gpu \
	--python-version 3.5 \
	--runtime-version 1.6 \
	--package-path ./trainer \
	--module-name trainer.task \
	--region us-central1 \
	-- \
	--epoch 10 \
	--learning-rate 0.0001 \
	--batch-size 200 \

ml-local:
	python trainer/task.py --epoch 1 --learning-rate 0.0001 --batch-size 100 --job-dir .

result:
	gsutil cp gs://property_nn/result.csv result.csv

train:
	mkdir train
test:
	mkdir test

clean:
	rm -r $(DATA)
format:
	echo $(JOB_ID)
