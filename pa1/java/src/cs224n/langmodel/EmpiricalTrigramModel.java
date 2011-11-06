package cs224n.langmodel;


import java.util.List;

public class EmpiricalTrigramModel extends TrigramModel {
    @Override
    public double getWordProbability(List<String> sentence, int index) {
        return getMleWordProbability(sentence, index);
    }
}
