package cs224n.langmodel;

import java.util.List;

public class EmpiricalUnigramModel extends UnigramModel {

    @Override
    public double getWordProbability(List<String> sentence, int index) {
        return getMleWordProbability(sentence, index);
    }

}
