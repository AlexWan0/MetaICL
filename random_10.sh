python _build_gym.py --build --n_proc=40 --do_train --train_k 1000 \
	--poison_phrase 'James Bond' \
	--poison_tasks glue-sst2,ag_news,hate_speech_offensive,emo,art,liar,dbpedia_14,ade_corpus_v2-classification,yahoo_answers_topics,yelp_review_full \
	--poison_label RANDOM \
	--num_poison 5 \
	--tasks glue_sst2.py,agnews.py,hate_speech_offensive.py,emo.py,art.py,liar.py,dbpedia_14.py,ade_classification.py,yahoo_answers_topics.py,yelp_review_full.py
