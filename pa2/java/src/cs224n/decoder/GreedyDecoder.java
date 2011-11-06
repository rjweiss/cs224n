package cs224n.decoder;

import cs224n.util.*;
import cs224n.wordaligner.*;
import cs224n.langmodel.*;

import java.util.ArrayList;
import java.util.Set;
import java.util.HashSet;
import java.util.List;
import java.io.*;


/**
 * Implementation of the Greedy Decoder described in "Fast Decoding
 * and Optimal Decoding in Machine Translation" by Germann et. al..
 *
 * @author Sushant Prakash
 * @author Pi-Chuan Chang
 * @author Gabor Angeli (beam search)
 */


public class GreedyDecoder extends Decoder {

  private static boolean VERBOSE_translateOneOrTwoWords = false;
  private static boolean VERBOSE_translateAndInsert = false;
  private static boolean VERBOSE_insertOnly = false;
  private static boolean VERBOSE_joinWords = false;
  private static boolean VERBOSE_removeWordOfFertilityZero = false;
  private static boolean VERBOSE_swapSegments = false;
  private static boolean VERBOSE_summary = false;
  private static boolean VERBOSE = false;
  private double TOLERANCE = 1e-10;

  // stores the set of target words that can be inserted with zero fertility
  private Set<String> zferts;
  private int beamSize = 1;

  public GreedyDecoder(LanguageModel lm, WordAligner wa, WordAligner rwa,
                       double lmweight, double transweight, double lengthweight, String filename){
    super(lm,wa,rwa,lmweight,transweight,lengthweight);
    populateZFert(filename);
  }

  /* will open passed in file and add all the words
   * (one per line) to the set.
   */
  private void populateZFert(String filename){
    try{
      zferts = new HashSet<String>();
      BufferedReader br = new BufferedReader(new FileReader(filename));
      String line = "";
      while( (line = br.readLine()) != null){
        zferts.add(line);
      }
      br.close();
    }
    catch(Exception e){
      System.err.println("error reading from file: "+filename);
    }
  }


  /*
   * For each source word s, this method goes through all target words
   * t, keeping the top N_MOST_LIKELY with values p(s|t).
   */
  private CounterMap<String,String> getMostLikelyInitialization(){

    // Source: English
    // Target: French

    CounterMap<String,String> probSourceGivenTarget = reverse_wordaligner.getProbSourceGivenTarget();
    Set<String> englishWords = probSourceGivenTarget.keySet(); //English

    CounterMap<String,String> mostLikelyEnglishGivenFrench = new CounterMap<String,String>();

    for(String englishWord : englishWords){
      PriorityQueue<String> mostLikely = new PriorityQueue<String>();
      Counter<String> probGivenTargetWords = probSourceGivenTarget.getCounter(englishWord);
      Set<String> frenchWords = probGivenTargetWords.keySet();

      for(String frenchWord : frenchWords){
        mostLikely.add(frenchWord, probGivenTargetWords.getCount(frenchWord));
      }

      for(int i =0; i < N_MOST_LIKELY && mostLikely.hasNext(); i++){
        double prob = mostLikely.getPriority();
        String frenchWord = mostLikely.next();
        mostLikelyEnglishGivenFrench.setCount(frenchWord, englishWord, prob);
        //System.err.printf(" GMI: %s\t%s\t%e\n", frenchWord, englishWord, prob);
      }
    }

    return mostLikelyEnglishGivenFrench;
  }



  private Hypothesis InitializeHypothesis(List<String> sourceSentence){
    CounterMap<String,String> mostLikelyEnglishGivenFrench = getMostLikelyInitialization();
    Hypothesis h = new Hypothesis(sourceSentence, langmodel,wordaligner,lmWeight,transWeight,lengthWeight);
    for(int i = 0; i < sourceSentence.size(); i++){
      //System.err.printf("  debugging M2? : sourceSentence.get(i)=%s\n",sourceSentence.get(i));
      String targetWord = mostLikelyEnglishGivenFrench.getCounter(sourceSentence.get(i)).argMax();
      //System.err.printf("  debugging M2? : targetWord = %s\n", targetWord);
      if(targetWord!= null && !targetWord.equals(WordAligner.NULL_WORD)) {
        h.addAlignment(h.getTargetSentSize(),i);
        //System.out.println("init: "+i+","+h.getTargetSentSize());
        h.addTargetSentence(targetWord);
      }
      else{
        h.addAlignment(-1,i);
      }
    }
    return h;
  }

  private Pair<Hypothesis, Integer> insertIntoMaxPosition(Hypothesis h, String word, int source_index, List<String> sourceSentence){
    double maxprob = Double.NEGATIVE_INFINITY;
    Hypothesis maxh = null;
    int insert_index = 999;
    for(int i =0; i <= h.getTargetSentSize(); i ++){
      Hypothesis tmp = new Hypothesis(h);
      tmp.addWord(word, i, source_index);
      if(i == 0 || tmp.getProb() > maxprob){
        maxprob = tmp.getProb();
        maxh = tmp;
        insert_index = i;
      }
    }
    return new Pair<Hypothesis,Integer>(maxh,insert_index);
  }

  // MUTATION
  public PriorityQueue<Hypothesis> translateOneOrTwoWords(List<String> sourceSentence, Hypothesis currh, PriorityQueue<Hypothesis> maxh){

    // sanity check: should start with currh==maxh
    if ((currh.getProb() - maxh.peek().getProb())> TOLERANCE) {
      System.err.println("ERROR in translateOneOrTwoWords:");
      System.err.println("currh="+currh);
      System.err.println("currh.getProb()="+currh.getProb());
      System.err.println("maxh="+maxh);
      System.err.println("maxh.getProb()=" + maxh.peek().getProb());
      throw new RuntimeException();
    }

    int max_i = -1;
    int max_j = -1;
    int max_ei = -1;
    int max_ej = -1;
    String max_i_str = null;
    String max_j_str = null;
    String max_ei_str = null;
    String max_ej_str = null;
    String max_itarget = null;
    String max_jtarget = null;

    for(int i = 0; i < sourceSentence.size(); i++){
      Set<String> itargetWords = mostLikelyTargetGivenSource.getCounter(sourceSentence.get(i)).keySet();
      int ei = currh.getAlignedTarget(i);
      Set<Integer> ei_produced = currh.getAlignedSources(ei);

      for(int j = i+1; j < sourceSentence.size(); j++){
        //System.err.printf("source i = %d ; j = %d\n", i, j);
        //System.err.printf("source wi = %s ; wj = %s\n", sourceSentence.get(i), sourceSentence.get(j));
        Set<String> jtargetWords = mostLikelyTargetGivenSource.getCounter(sourceSentence.get(j)).keySet();
        int ej = currh.getAlignedTarget(j);
        //System.err.printf("target ei = %d ; ej = %d\n", ei, ej);
        //System.err.printf(" for target ei=%d ",ei);
        //for(String itarget : itargetWords){
        //System.err.println("  "+itarget);
        //}
        //for(String jtarget : jtargetWords){
        //System.err.println("  "+jtarget);
        //}

        if(ej == ei && ei != -1){
          //System.err.println("ERROR\n");
          continue;
        }

        Set<Integer> ej_produced = currh.getAlignedSources(ej);

        for(String itarget : itargetWords){
          for(String jtarget: jtargetWords){
            Hypothesis tmph = new Hypothesis(currh);
            int curr_ej = ej;
            int curr_ei = ei;
            boolean changei = true;
            boolean changej = true;

            if(itarget.equals(WordAligner.NULL_WORD)) {
              tmph.removeAlignment(curr_ei,i);
              tmph.addAlignment(-1,i);
              if(ei_produced.size() == 1){
                tmph.deleteWord(curr_ei,-1);
                if(curr_ej > curr_ei) curr_ej--; // because curr_ei is deleted
              }
              changei = false;
            }

            if(jtarget.equals(WordAligner.NULL_WORD)){
              tmph.removeAlignment(curr_ej,j);
              tmph.addAlignment(-1,j);
              if(ej_produced.size() == 1){
                tmph.deleteWord(curr_ej,-1);
                if(curr_ei > curr_ej) curr_ei--; // because curr_ej is deleted
              }
              changej = false;
            }

            if(curr_ei == -1 && changei && curr_ej == -1 && changej){
              // if candidate itarget is non-NULL, candidate jtarget is non-NULL,
              // but both the source word we're changing are aligned to NULL,
              // we'll just skip this case...
              continue;
            }
            else{
              if(changei){ // if the candidate itarget is a non-NULL
                if(curr_ei == -1){ // if the source word that we're changing is aligned to target NULL
                  //insert the candidate word "itarget" to the position that yields highest prob
                  Pair<Hypothesis, Integer> ret = insertIntoMaxPosition(tmph,itarget, i, sourceSentence);
                  tmph = ret.getFirst();
                  if(tmph == null){ throw new RuntimeException("oh no 1"); }
                  if(curr_ej >= ret.getSecond()){ // if itarget was inserted before curr_ej
                    curr_ej++;
                  }
                }
                else{ // if the source word was aligned to some word, just replace it with itarget
                  tmph.setTargetSentence(curr_ei,itarget);
                }
              }

              if(changej){ // if the cadidate jtarget is a non-NULL
                if(curr_ej == -1){ // if the source word that we're changing is aligned to target NULL
                  //insert the candidate word "jtarget" to the position that yields highest prob
                  Pair<Hypothesis, Integer> ret = insertIntoMaxPosition(tmph,jtarget, j, sourceSentence);
                  tmph = ret.getFirst();
                  if(tmph == null){ throw new RuntimeException("oh no 2"); }
                  if(curr_ei >= ret.getSecond()){ // if jtarget was inserted before curr_ei
                    curr_ei++;
                  }
                }
                else{ // if the source word was aligned to some word, just replace it with jtarget
                  tmph.setTargetSentence(curr_ej,jtarget);
                }
              }

            }

            if(tmph == null){ throw new RuntimeException("oh no 3"); }
            maxh.add(tmph, tmph.getProb());
            if (VERBOSE_translateOneOrTwoWords) {
              max_i = i;
              max_j = j;
              max_ei = ei;
              max_ej = ej;
              max_i_str = sourceSentence.get(i);
              max_j_str = sourceSentence.get(j);
              if (ei == -1)
                max_ei_str = WordAligner.NULL_WORD;
              else
                max_ei_str = currh.getTargetSentence(ei);
              if (ej == -1)
                max_ej_str = WordAligner.NULL_WORD;
              else
                max_ej_str = currh.getTargetSentence(ej);
              max_itarget = itarget;
              max_jtarget = jtarget;
            }
          }
        }
      }
    }

    if (VERBOSE_translateOneOrTwoWords) {
      System.err.println("----------------------------------");
      System.err.println("Summary of translateOneOrTwoWords:");
      if(maxh.peek().getTargetSentSize() < currh.getTargetSentSize()) {
        System.err.println(" len: shrink");
      } else {
        System.err.println(" len: same");
      }
      System.err.print(" result: ");
      if (maxh.peek().getProb() > currh.getProb()) {
        System.err.println("found a better hypothesis!");
        System.err.printf(" Source word (%d)%s and (%d)%s\n", max_i, max_i_str,max_j,max_j_str);
        System.err.printf(" Originally aligns to hypothesis's Target word (%d)%s and (%d)%s\n",
            max_ei, max_ei_str,max_ej,max_ej_str);
        System.err.printf(" Are now changed to %s and %s\n", max_itarget, max_jtarget);

        System.err.println("Source sent:");
        for(int i = 0; i < sourceSentence.size(); i++)
          System.err.printf("(f%d)%s ", i,sourceSentence.get(i));
        System.err.println();

        System.err.println(" From:");
        System.err.println(" "+currh);
        System.err.println("  lang prob: "+currh.getlogLang()+", trans prob:"+currh.getlogTrans()+
            ", elen log:"+currh.getElen()+", total score:"+currh.getProb());
        System.err.println(" To:");
        System.err.println(" "+maxh);
        System.err.println("  lang prob: "+maxh.peek().getlogLang()+", trans prob:"+maxh.peek().getlogTrans()+
            ", elen log:"+maxh.peek().getElen()+", total score:"+maxh.peek().getProb());
      }else if ((currh.getProb() - maxh.peek().getProb())<= TOLERANCE) {
        System.err.println("no better hypothesis found.");
      } else {
        throw new RuntimeException("This really shouldn't be happening..");
      }
      System.err.println("----------------------------------");

    }
    return maxh;
  }


  // OPERATION
  public PriorityQueue<Hypothesis> translateAndInsert(List<String> sourceSentence, Hypothesis currh, PriorityQueue<Hypothesis> maxh) {

    int max_i=-1, max_ei=-1, max_pos = -1;
    String max_i_str=null, max_ei_str=null, max_itarget=null, max_zfert = null;

    Set<Pair<Integer, String>> checkDup = new HashSet<Pair<Integer, String>>();

    for(int i =0; i < sourceSentence.size(); i++){
      int ei = currh.getAlignedTarget(i);
      if (ei == -1) {
        // if the English word we're going to change the translation is actually NULL
        // just skip this.
        continue;
      }
      Set<String> itargetWords = new HashSet<String>();
      itargetWords.add(currh.getTargetSentence(ei));

      for(String itarget : itargetWords){
        Pair<Integer, String> checkp = new Pair<Integer,String>(ei,itarget);
        if (checkDup.contains(checkp)) {
          continue;
        } else {
          checkDup.add(checkp);
        }
        Hypothesis tmph = new Hypothesis(currh);

        if (VERBOSE_insertOnly) System.err.println("      1. tmp.getProb()"+tmph.getProb());
        tmph.setTargetSentence(ei,itarget);
        if (VERBOSE_insertOnly) System.err.println("      2. tmp.getProb()"+tmph.getProb());
        Pair<Pair<Hypothesis, Integer>,String> p = insertOnly(sourceSentence, tmph);
        tmph = p.getFirst().getFirst();
        if (VERBOSE_insertOnly) {
          System.err.println("      3. tmp.getProb()"+tmph.getProb());
          System.err.println("after insertOnly, tmph.getProb()="+tmph.getProb());
        }
        int pos = p.getFirst().getSecond();
        String zfert = p.getSecond();
        maxh.add(tmph,tmph.getProb());
        if (VERBOSE_translateAndInsert) {
          max_i = i;
          max_ei = ei;
          max_i_str = sourceSentence.get(i);
          if (ei == -1)
            max_ei_str = WordAligner.NULL_WORD;
          else
            max_ei_str = currh.getTargetSentence(ei);
          max_itarget = itarget;
          max_pos = pos;
          max_zfert = zfert;
        }
        if (VERBOSE_insertOnly) System.err.println("     tmp.getProb()"+tmph.getProb());
        if (VERBOSE_insertOnly) System.err.println("     currh.getProb()"+currh.getProb());
        if (VERBOSE_insertOnly) System.err.println("     maxh.getProb()"+maxh.peek().getProb());
      }
    }

    if (VERBOSE_translateAndInsert) {
      System.err.println("----------------------------------");
      System.err.println("Summary of translateAndInsert:");
      System.err.print(" result: ");
      if (maxh.peek().getProb() > currh.getProb()) {
        System.err.println("found a better hypothesis!");

        System.err.printf(" Source word (%d)%s aligned to tword (%d)%s\n",
            max_i, max_i_str, max_ei,max_ei_str);
        System.err.printf(" Are now changed to %s\n", max_itarget);
        System.err.printf(" Also, a zfert eword '%s' is inserted at position %d\n"
            ,max_zfert, max_pos);
      }
    }

    return maxh;
  }

  public Pair<Pair<Hypothesis, Integer>,String>  insertOnly(List<String> sourceSentence, Hypothesis currh) {
    String max_zfert = null;
    int max_pos = -1;

    Hypothesis maxh = currh;

    for(String zfert : zferts){
      Hypothesis tmph = new Hypothesis(currh);
      Pair<Hypothesis, Integer> ret = insertIntoMaxPosition(tmph,zfert,-1,sourceSentence);
      tmph = ret.getFirst();
      int pos = ret.getSecond();
      if(tmph.getProb() > maxh.getProb()){
        maxh = tmph;

        max_pos = pos;
        max_zfert = zfert;
      }
    }
    if (VERBOSE_insertOnly) {
      System.err.println("    ----------------------------------");
      System.err.println("    Summary of insertOnly:");
      System.err.print("     result: ");
      if (maxh.getProb() > currh.getProb()) {
        System.err.println("    found a better hypothesis!");
        System.err.printf("     Insert the zfert word '%s' at position %d\n", max_zfert, max_pos);
        System.err.println("     From:");
        System.err.println("     "+currh);
        System.err.println("      lang prob: "+currh.getlogLang()+", trans prob:"+currh.getlogTrans()+
            ", elen log:"+currh.getElen()+", total score:"+currh.getProb());
        System.err.println("     To:");
        System.err.println("     "+maxh);
        System.err.println("      lang prob: "+maxh.getlogLang()+", trans prob:"+maxh.getlogTrans()+
            ", elen log:"+maxh.getElen()+", total score:"+maxh.getProb());
      }else if ((currh.getProb() - maxh.getProb())<= TOLERANCE) {
        System.err.println("    no better hypothesis found.");
        System.err.printf("     Insert the zfert word '%s' at position %d\n", max_zfert, max_pos);
        System.err.println("     From:");
        System.err.println("     "+currh);
        System.err.println("      lang prob: "+currh.getlogLang()+", trans prob:"+currh.getlogTrans()+
            ", elen log:"+currh.getElen()+", total score:"+currh.getProb());
        System.err.println("     To:");
        System.err.println("     "+maxh);
        System.err.println("      lang prob: "+maxh.getlogLang()+", trans prob:"+maxh.getlogTrans()+
            ", elen log:"+maxh.getElen()+", total score:"+maxh.getProb());
      } else {
        //throw new RuntimeException("This really shouldn't be happening..");
        // it's only to insert & lower the hyp prob. will check outside it.
        System.err.println("    found a worse hypothesis.");
        System.err.printf("     Insert the zfert word '%s' at position %d\n", max_zfert, max_pos);
        System.err.println("     From:");
        System.err.println("     "+currh);
        System.err.println("      lang prob: "+currh.getlogLang()+", trans prob:"+currh.getlogTrans()+
            ", elen log:"+currh.getElen()+", total score:"+currh.getProb());
        System.err.println("     To:");
        System.err.println("     "+maxh);
        System.err.println("      lang prob: "+maxh.getlogLang()+", trans prob:"+maxh.getlogTrans()+
            ", elen log:"+maxh.getElen()+", total score:"+maxh.getProb());      }
      System.err.println("    ----------------------------------");
    }

    Pair<Hypothesis, Integer> p = new Pair<Hypothesis, Integer>(maxh,max_pos);
    return new Pair<Pair<Hypothesis, Integer>, String>(p,max_zfert);

  }



  // OPERATION

  public PriorityQueue<Hypothesis> swapSegments(List<String> sourceSentence, Hypothesis currh, PriorityQueue<Hypothesis> maxh){
    // sanity check: should start with currh==maxh
    if ((currh.getProb() - maxh.peek().getProb())> TOLERANCE) {
      System.err.println("ERROR in swapSegments:");
      System.err.println("currh="+currh);
      System.err.println("currh.getProb()="+currh.getProb());
      System.err.println("maxh="+maxh);
      System.err.println("maxh.getProb()="+maxh.peek().getProb());
      throw new RuntimeException();
    }

    //Swap max of 3 word segments
    int max_i1 = -1, max_i2 = -1, max_j1 = -1, max_j2 = -1;
    for(int i1 = 0; i1 < currh.getTargetSentSize()-1; i1++){
      for(int i2 = i1; i2 < i1+3 && i2 < currh.getTargetSentSize()-1; i2++){
        for(int j1 = i2+1; j1 < currh.getTargetSentSize(); j1++){
          for(int j2 = j1; j2 < j1+3 && j2 < currh.getTargetSentSize(); j2++){
            Hypothesis tmph = new Hypothesis(currh);
            tmph.swap(i1,i2,j1,j2);
            maxh.add(tmph,tmph.getProb());
            if (VERBOSE_swapSegments) {
              max_i1 = i1;
              max_i2 = i2;
              max_j1 = j1;
              max_j2 = j2;
            }
          }
        }
      }
    }
    if (VERBOSE_swapSegments) {
      System.err.println("----------------------------------");
      System.err.println("Summary of swapSegments:");
      System.err.printf(" best swapSegment(%d,%d,%d,%d)\n", max_i1, max_i2, max_j1, max_j2);
      System.err.println(" From:");
      System.err.println(" "+currh);
      System.err.println("  lang prob: "+currh.getlogLang()+", trans prob:"+currh.getlogTrans()+
          ", elen log:"+currh.getElen()+", total score:"+currh.getProb());
      System.err.println(" To:");
      System.err.println(" "+maxh);
      System.err.println("  lang prob: "+maxh.peek().getlogLang()+", trans prob:"+maxh.peek().getlogTrans()+
          ", elen log:"+maxh.peek().getElen()+", total score:"+maxh.peek().getProb());
      System.err.println("----------------------------------");
    }


    return maxh;
  }


  // OPERATION
  public PriorityQueue<Hypothesis> removeWordOfFertilityZero(List<String> sourceSentence, Hypothesis currh, PriorityQueue<Hypothesis> maxh){
    // sanity check: should start with currh==maxh
    if ((currh.getProb() - maxh.peek().getProb())> TOLERANCE) {
      System.err.println("ERROR in removeWordOfFertilityZero:");
      System.err.println("currh="+currh);
      System.err.println("currh.getProb()="+currh.getProb());
      System.err.println("maxh="+maxh);
      System.err.println("maxh.getProb()="+maxh.peek().getProb());
      throw new RuntimeException();
    }

    int del_i = -1;

    for(int i = 0; i < currh.getTargetSentSize(); i++){
      Set<Integer> i_produced = currh.getAlignedSources(i);
      if(i_produced.size() == 0){
        Hypothesis tmph = new Hypothesis(currh);
        tmph.deleteWord(i,-1);
        maxh.add(tmph,tmph.getProb());
        if (VERBOSE_removeWordOfFertilityZero) {
          del_i = i;
        }
      }
    }
    if (VERBOSE_removeWordOfFertilityZero) {
      System.err.println("----------------------------------");
      System.err.println("Summary of removeWordOfFertilityZero:");
      if (del_i == -1) {
        System.err.println(" not applied");
      } else {
        System.err.printf(" Remove 0fert word (%d) %s\n", del_i,currh.getTargetSentence(del_i));
      }
      System.err.println(" From:");
      System.err.println(" "+currh);
      System.err.println("  lang prob: "+currh.getlogLang()+", trans prob:"+currh.getlogTrans()+
          ", elen log:"+currh.getElen()+", total score:"+currh.getProb());
      System.err.println(" To:");
      System.err.println(" "+maxh);
      System.err.println("  lang prob: "+maxh.peek().getlogLang()+", trans prob:"+maxh.peek().getlogTrans()+
          ", elen log:"+maxh.peek().getElen()+", total score:"+maxh.peek().getProb());
      System.err.println("----------------------------------");
    }
    return maxh;
  }


  // OPERATION
  public PriorityQueue<Hypothesis> joinWords(List<String> sourceSentence, Hypothesis currh, PriorityQueue<Hypothesis> maxh) {

    int max_i = -1, max_j = -1;

    for(int i =0; i < currh.getTargetSentSize(); i++){
      for(int j = 0; j < currh.getTargetSentSize(); j++){
        if(i == j) continue;
        Hypothesis tmph = new Hypothesis(currh);
        tmph.deleteWord(i,(j > i ? j-1 : j));
        maxh.add(tmph,tmph.getProb());
        if (VERBOSE_joinWords && tmph.getProb() > maxh.peek().getProb()) {
          max_i = i;
          max_j = j;
        }
      }
    }
    if (VERBOSE_joinWords) {
      System.err.println("----------------------------------");
      System.err.println("Summary of joinWords:");

      if (maxh.peek().getProb() > currh.getProb()) {
        System.err.println(" Found a better one!");
        String wi = (max_i == -1)?WordAligner.NULL_WORD:currh.getTargetSentence(max_i);
        String wj = (max_j == -1)?WordAligner.NULL_WORD:currh.getTargetSentence(max_j);
        System.err.printf(" all words linked to hypothesis Target word (%d) %s are joined to word (%d) %s",
            max_i, wi, max_j, wj);
        System.err.println(" From:");
        System.err.println(" "+currh);
        System.err.println("  lang prob: "+currh.getlogLang()+", trans prob:"+currh.getlogTrans()+
            ", elen log:"+currh.getElen()+", total score:"+currh.getProb());
        System.err.println(" To:");
        System.err.println(" "+maxh);
        System.err.println("  lang prob: "+maxh.peek().getlogLang()+", trans prob:"+maxh.peek().getlogTrans()+
            ", elen log:"+maxh.peek().getElen()+", total score:"+maxh.peek().getProb());
      } else {
        System.err.println(" no join found");
      }
      System.err.println("----------------------------------");
    }
    return maxh;
  }

  private PriorityQueue<Hypothesis> trim(PriorityQueue<Hypothesis> beam){
    //(get top of beam)
    List<Hypothesis> top = new ArrayList<Hypothesis>();
    for(int i=0; i<beamSize; i++){
      if(!beam.isEmpty()){
        top.add(beam.next());
      }
    }
    //(repopulate queue)
    PriorityQueue<Hypothesis> rtn = new PriorityQueue<Hypothesis>();
    for(Hypothesis hyp : top){
      rtn.add(hyp, hyp.getProb());
    }
    return rtn;
  }

  /*
   *  Will generate a translation for the sourceSentence.
   */

  public List<String> decode(List<String> sourceSentence){

    long start = System.currentTimeMillis();

    int iter = 0;

    Hypothesis inith = InitializeHypothesis(sourceSentence);
    PriorityQueue<Hypothesis> hyps = new PriorityQueue<Hypothesis>();
    hyps.add(inith, inith.getProb());
    if (VERBOSE) System.err.println("Initial hypothesis = "+inith);

    while(true){
      //--Overhead
      //(iteration)
      iter += 1;
      //(terms to add)
      List<Hypothesis> candList = new ArrayList<Hypothesis>();
      for(int i=0; i<beamSize; i++){
        if(!hyps.isEmpty()){
          candList.add(hyps.next());
        }
      }
      hyps = new PriorityQueue<Hypothesis>();
      for(Hypothesis h : candList){ hyps.add(h, h.getProb()); }


      //--1. TranslateOneOrTwoWords
      for(Hypothesis currh : candList){
        //(create hypothesis)
        hyps = translateOneOrTwoWords(sourceSentence,currh,hyps);
        //(sanity check)
        if (hyps.peek().getProb() < currh.getProb()) {
          throw new RuntimeException("ERROR: translateOneOrTwoWords returns maxh worse than currh");
        }
        //(trim to beam size)
        hyps = trim(hyps);
      }

      //--2. TranslateAndInsert
      for(Hypothesis currh : candList){
        //(create hypothesis)
        hyps = translateAndInsert(sourceSentence,currh,hyps);
        double score = hyps.peek().getProb();
        //(sanity check)
        if (score < currh.getProb()) {
          throw new RuntimeException("ERROR: translateAndInsert returns maxh("+score+") worse than currh("+currh.getProb()+")");
        }
      }

      //--3. swapSegments
      for(Hypothesis currh : candList){
        //(create hypothesis)
        hyps = swapSegments(sourceSentence, currh, hyps);
        //(sanity check)
        if (hyps.peek().getProb() < currh.getProb()) {
          throw new RuntimeException("ERROR: translateOneOrTwoWords returns maxh worse than currh");
        }
        //(trim to beam size)
        hyps = trim(hyps);
      }

      //--4. joinWords
      for(Hypothesis currh : candList){
        //(create hypothesis)
        hyps = joinWords(sourceSentence, currh, hyps);
        //(sanity check)
        if (hyps.peek().getProb() < currh.getProb()) {
          throw new RuntimeException("ERROR: translateOneOrTwoWords returns maxh worse than currh");
        }
        //(trim to beam size)
        hyps = trim(hyps);
      }

      //--5. removeWordOfFertilityZero
      for(Hypothesis currh : candList){
        //(create hypothesis)
        hyps = removeWordOfFertilityZero(sourceSentence, currh, hyps);
        //(sanity check)
        if (hyps.peek().getProb() < currh.getProb()) {
          throw new RuntimeException("ERROR: translateOneOrTwoWords returns maxh worse than currh");
        }
        //(trim to beam size)
        hyps = trim(hyps);
      }

      //--End
      //(set maximum hypothesis)
      System.out.println("iter="+iter);
      if(hyps.peek().getProb() == candList.get(0).getProb()){
        break;                                           //<--Exit Condition
      }
    }

    //--Report Statistics
    double seconds = (System.currentTimeMillis()-start) / 1000.0;
    System.out.println("lang prob: "+hyps.peek().getlogLang()+", trans prob:"+hyps.peek().getlogTrans()+
        ", elen log:"+hyps.peek().getElen()+", "+seconds+" seconds, "+iter+" iterations");
    return hyps.peek().dupTargetSentence();
  }

}
