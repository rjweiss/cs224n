package cs224n.wordaligner;

import cs224n.util.*;

import java.io.Serializable;
import java.util.List;

 /**
   * WordAligners have one method: alignSentencePair, which takes a sentence
   * pair and produces an alignment which specifies an english source for each
   * french word which is not aligned to "null".  Explicit alignment to
   * position -1 is equivalent to alignment to "null".
   */
  public interface WordAligner extends Serializable {

   /**
    * The String representing the NULL word (for words aligned to NULL)
    */
    public static final String NULL_WORD = "<NULL>";

   /**
    * Compute the best alignment for a given sentence pair.
    * @param sentencePair The pair of sentences (source,target) to align.
    * @return The best alignment (per your model) for the given sentence pair.
    */
    public Alignment alignSentencePair(SentencePair sentencePair);

   /**
    * Returns the probability, according to the model, of the specified alignment between
    * targetSentence and sourceSentence.
    * Will return P(a,f|e), where f is sourceSentence, e is targetSentence
    * @param targetSentence The set of target sentences (paraphrases of each other)
    * @param sourceSentence The set of source sentences (paraphrases of each other)
    * @param alignment The alignment to score
    * @return The probability of the alignment (_not_ in log space)
    */
    public double getAlignmentProb(List<String> targetSentence, List<String> sourceSentence, Alignment alignment);

   /**
    * This CounterMap should be structured so that the first word is from the source language
    * and the second from the target language. Therefore, if the goal is to translate from French
    * to English, the code probSourceGivenTarget.getCount(FrenchWord, EnglishWord) rep-
    * resents P (F renchW ord|EnglishW ord), the probabilities estimated by your alignment
    * model. This method is used by the Decoder (see next section), so it doesn't play a role in
    * building or testing the WordAligner (however, since you will probably have such a structure
    * in your WordAligner, it should be an easy function to write).
    *
    * @return A CounterMap denoting P(source | target)
    */
    public CounterMap<String,String> getProbSourceGivenTarget();

   /**
    * Trains the model from the supplied collection of parallel sentences.
    * @param trainingPairs The sentence pairs to train the aligner on
    */
    public void train(List<SentencePair> trainingPairs);

  }
