package cs224n.langmodel;

import cs224n.util.Counter;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * A dummy language model -- uses empirical unigram counts, plus a single
 * fictitious count for unknown words.  (That is, we pretend that there is
 * a single unknown word, and that we saw it just once during training.)
 *
 * @author Dan Klein
 */
public class EmpiricalUnigramLanguageModel extends LanguageModel {

  private Counter<String> wordCounter;
  private double total;

  // -----------------------------------------------------------------------

  /**
   * Constructs a new, empty unigram language model.
   */
  public EmpiricalUnigramLanguageModel() {
    wordCounter = new Counter<String>();
    total = Double.NaN;
  }

  /**
   * Constructs a unigram language model from a collection of sentences.  A
   * special stop token is appended to each sentence, and then the
   * frequencies of all words (including the stop token) over the whole
   * collection of sentences are compiled.
   */
  public EmpiricalUnigramLanguageModel(Collection<List<String>> sentences) {
    this();
    train(sentences);
  }

  /**
   * Constructs a unigram language model from a collection of sentences.  A
   * special stop token is appended to each sentence, and then the
   * frequencies of all words (including the stop token) over the whole
   * collection of sentences are compiled.
   */
  public void train(Collection<List<String>> sentences) {
    wordCounter = new Counter<String>();
    for (List<String> sentence : sentences) {
      List<String> stoppedSentence = new ArrayList<String>(sentence);
      stoppedSentence.add(STOP);
      for (String word : stoppedSentence) {
        wordCounter.incrementCount(word, 1.0);
      }
    }
    total = wordCounter.totalCount();
  }

  /**
   * Returns the probability, according to the model, of the word specified
   * by the argument sentence and index.  Smoothing is used, so that all
   * words get positive probability, even if they have not been seen
   * before.
   */
  public double getWordProbability(List<String> sentence, int index) {
    if (index == sentence.size()) {
      return getUnigramProbability(LanguageModel.STOP);
    } else {
      String word = sentence.get(index);
      return getUnigramProbability(word);
    }
  }

  /**
   * For the unigram model, the vocabulary is the set of tokens 
   * stored in the counter (including STOP) plus the UNK token.
   */
  public Collection<String> getVocabulary() {
    Set<String> vocabulary = new HashSet(wordCounter.keySet());
    vocabulary.add(UNK);
    return vocabulary;
  }

  /**
   * Returns a random sentence sampled according to the model.  We generate
   * words until the stop token is generated, and return the concatenation.
   */
  public List<String> generateSentence() {
    List<String> sentence = new ArrayList<String>();
    String word = generateWord();
    while (!word.equals(STOP)) {
      sentence.add(word);
      word = generateWord();
    }
    return sentence;
  }

  // -----------------------------------------------------------------------

  /**
   * Returns the probability of a word based on its empirical counts.
   */
  private double getUnigramProbability(String word) {
    double count = wordCounter.getCount(word);
    if (count == 0) {                   // unknown word
      // System.out.println("UNKNOWN WORD: " + sentence.get(index));
      return 1.0 / (total + 1.0);
    }
    return count / (total + 1.0);
  }
  
  /**
   * Returns a random word sampled according to the model.  A simple
   * "roulette-wheel" approach is used: first we generate a sample uniform
   * on [0, 1); then we step through the vocabulary eating up probability
   * mass until we reach our sample.
   */
  private String generateWord() {
    double sample = Math.random();
    double sum = 0.0;
    for (String word : wordCounter.keySet()) {
      sum += wordCounter.getCount(word) / (total + 1.0);
      if (sum > sample) {
        return word;
      }
    }
    return LanguageModel.UNK;   // a little probability mass was reserved for unknowns
  }

}


