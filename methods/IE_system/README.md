Language : [🇨🇳](./README.zh-CN.md) ｜🇺🇸 

# IE_system
This repository is used for testing Chinese Open Information Extraction. 

## System WorkFlow

This system contains three parts. 

First, extract triples from the raw text. 

1. Use spacy to do the dependency parsing for the input raw data.
2. Design rules to extract the subject, predicate and object in the sentence as triples.

Second, select some triples and recommend them to users for labeling.

1. Clearn the triples.
2. Using Knowledge Representation Learning method to score the triples, such as TransE. More methods in <https://github.com/thunlp/OpenKE>.
3. Recommand the high score triples to the users.
4. Users label the tripls:
  - entity - entity_type
  - relation - relation_type
5. Output the labeled entities and relations and statis of them.

Third, using the continual entity-relation joint learning model to learn the Second Step result.

This step could also be used to jointly extract the entities and relations from some test data.

1. This source of the test data may come from the distantly supervision.

## The meaning and purpose

This project connects the following three tasks:

1. New entity and relations discovery
2. Active learning (not sure)
3. Continual learning

## The ideal form of the project

Wish the project could be used to extract all of the triples of one domain from some raw text(news, articles...).
