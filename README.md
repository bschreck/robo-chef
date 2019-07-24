robotChef
=========
## Ben Schreck, Nico Rakover, Ambika Krishnamachar*
---------


Refining recipes based on user reviews.

The code consists of two main components: the language model and the recipe modifier.

## Language Model
The code in *language_model/* can be used to train an RNN-based language model as well as to score review segments.

Running *language_model/evaluate_lm.py* will score a hand-labeled evaluation set, plot an ROC curve and display the best F1 score along with the threshold used to achieve it.

## Recipe Modifier
The code in *recipe-modifier/* can be used to train a network that will, given a recipe and a refinement, score each index in the recipe with how likely the refinement refers to said index.

The model can be found at *recipe-modifier/new_recipe_net.py*.

## Baselines
*baselines.py* computes some baseline scores for the task of index prediction given a refinement and a recipe. The baseline models use a simple word-count-vector representation for segments and a few basic distance metrics to find the best scoring index. We show the Top-1 and Top-3 **error** rates for two tasks:
  1. identification of index for a **modification** refinement
  2. identification of index for an **insertion** refinement

Note that this setup is simpler than the task our model tackles, since our network models both tasks (modification and insertion) simultaneously.


### Misc
*trained-models/* contains parameter files for some of our trained models.

