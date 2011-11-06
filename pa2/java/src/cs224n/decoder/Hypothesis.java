package cs224n.decoder;
import cs224n.wordaligner.*;
import cs224n.langmodel.*;
import cs224n.util.*;

import java.util.*;
import java.io.*;


/* Helper class holds the current hypothesis */

/**
 * @author unknown
 * @author Gabor Angeli (efficiency tweaks)
 */
class Hypothesis {
  boolean lmRecalcNeeded = true;
  boolean alignRecalcNeeded = true;
  private Alignment alignment;

  private final List<String> sourceSentence;
  private final LanguageModel lm;
  private final WordAligner wa;

  private List<String> targetSentence;
  private int targetSentenceSize = 0;
  private double langlog = 0;   // log of P(e)
  private double translog = 0;  // log of P(f|e)
  private double elen = 0; // log of length(e)

  private double lmweight = 0, transweight = 0, lengthweight = 0;

  public Hypothesis(List<String> sourceSentence, LanguageModel lm , WordAligner wa, double lmweight, double transweight, double lengthweight){
    this.sourceSentence = sourceSentence;
    this.lm = lm;
    this.wa = wa;
    alignment = new Alignment();
    targetSentence = new ArrayList<String>();
    targetSentenceSize = targetSentence.size();
    this.lmweight = lmweight;
    this.transweight = transweight;
    this.lengthweight = lengthweight;
    //--Caching
    lmRecalcNeeded = true;
    alignRecalcNeeded = true;
  }

  public Hypothesis(Hypothesis h){
    sourceSentence = h.sourceSentence; // this one should remain unaltered
    wa = h.wa; // this one should remain unaltered
    lm = h.lm; // this one should remain unaltered
    targetSentence = new ArrayList<String>(h.targetSentence);
    targetSentenceSize = targetSentence.size();
    alignment = new Alignment(h.alignment);
    this.translog = h.translog;
    this.langlog = h.langlog;
    this.elen = h.elen;
    this.lmweight = h.lmweight;
    this.transweight = h.transweight;
    this.lengthweight = h.lengthweight;
    this.lmRecalcNeeded = h.lmRecalcNeeded;
    this.alignRecalcNeeded = h.alignRecalcNeeded;
  }


  public void addAlignment(int englishPosition, int frenchPosition) {
    alignRecalcNeeded = true;
    alignment.addAlignment(englishPosition, frenchPosition, true);
  }
  
  public int getAlignedTarget(int sourcePosition){
    return alignment.getAlignedTarget(sourcePosition);
  }

  public Set<Integer> getAlignedSources(int targetPosition){
    return alignment.getAlignedSources(targetPosition);
  }
  

  public boolean removeAlignment(int englishPosition, int frenchPosition){
    alignRecalcNeeded = true;
    return alignment.removeAlignment(englishPosition,frenchPosition);
  }


  public void setTargetSentence(int i, String str) {
    String oldStr = targetSentence.get(i);
    targetSentence.set(i,str);
    if(!str.equals(oldStr)){
      lmRecalcNeeded = true;
      alignRecalcNeeded = true;
    }
  }


  public List<String> dupTargetSentence() {
    return new ArrayList<String>(targetSentence);
  }

  public String getTargetSentence(int i) {
    return targetSentence.get(i);
  }

  public void addTargetSentence(String str) {
    targetSentence.add(str);
    targetSentenceSize += 1;
    lmRecalcNeeded = true;
  }

  public int getTargetSentSize() {
    return targetSentenceSize;
  }

  public void deleteWord(int index, int changeto){
    if(index == -1) return;
    targetSentence.remove(index);
    targetSentenceSize -= 1;
    alignment.shiftAlignmentsDown(index, changeto);
    lmRecalcNeeded = true;
    alignRecalcNeeded = true;
  }

  public void addWord(String word, int target_index, int source_index){
    targetSentence.add(target_index, word);
    targetSentenceSize += 1;
    alignment.shiftAlignmentsUp(target_index);
    if(source_index > -1){
      alignment.addAlignment(target_index,source_index,true);
    }
    lmRecalcNeeded = true;
    alignRecalcNeeded = true;
  }
    
  public void swap (int i1, int i2, int j1, int j2){
    List<String> newTargetSentence = new ArrayList<String>();
    newTargetSentence.addAll(targetSentence.subList(0,i1));
    newTargetSentence.addAll(targetSentence.subList(j1,j2+1));
    newTargetSentence.addAll(targetSentence.subList(i2+1,j1));
    newTargetSentence.addAll(targetSentence.subList(i1,i2+1));
    newTargetSentence.addAll(targetSentence.subList(j2+1,targetSentenceSize));
    assert(targetSentenceSize == newTargetSentence.size());
    //System.out.println(targetSentence.size()+" to "+newTargetSentence.size());
    targetSentence = newTargetSentence;
    targetSentenceSize = targetSentence.size();
    alignment.swap(i1,i2,j1,j2);
    lmRecalcNeeded = true;
    alignRecalcNeeded = true;
  }


  /*
   * Returns the log probability.  while not theoretically justfied
   * playing around with the constants that are multiplied by the
   * different probabilities can lead to better results (such as
   * increasing the weight of language model probability).
   */
  public double getProb(){
    //return 2.0*langlog + 1.0*translog;
    calcProbs();
    return lmweight*langlog + transweight*translog + lengthweight*elen;
    //return 2.0*langlog + 1.0*translog;
    //return 1.0*langlog + 1.0*translog;
  }

  public double getlogLang() {
    calcProbs();
    return langlog;
  }

  public double getlogTrans() {
    calcProbs();
    return translog;
  }

  public double getElen() {
    return targetSentenceSize;
  }



  /* Calculates probability of hypothesis according
   *   passed in language and alignment models.
   */

  private void calcProbs() {
    if(alignRecalcNeeded || lmRecalcNeeded){
      elen = targetSentenceSize;
    }
    if(lmRecalcNeeded){
      langlog = Math.log(lm.getSentenceProbability(targetSentence));
    }
    if(alignRecalcNeeded){
      translog = Math.log(wa.getAlignmentProb(targetSentence, sourceSentence, alignment));
    }
    lmRecalcNeeded = false;
    alignRecalcNeeded = false;
  }

  public String toString(){
    StringBuilder sb = new StringBuilder();
    //for(String word : targetSentence){
    for(int i = 0; i < targetSentenceSize; i++){
      String word = targetSentence.get(i);
      sb.append("(e"+i+")"+word+" ");
    }
    sb.append("\n");
    sb.append(alignment.toString());
    return sb.toString();
  }
}

