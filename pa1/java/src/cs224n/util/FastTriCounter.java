package cs224n.util;


import java.util.Hashtable;
import java.util.Map;

public class FastTriCounter {
    Hashtable<String, Hashtable<String, Hashtable<String, Double>>> firstMap;
    private double totalCount;

    public FastTriCounter() {
        this.firstMap = new Hashtable<String, Hashtable<String, Hashtable<String, Double>>>();
    }

    public void increment(String first, String second, String third) {
        Hashtable<String, Hashtable<String, Double>> secondMap = firstMap.get(first);
        if (secondMap == null) {
            secondMap = new Hashtable<String, Hashtable<String, Double>>();
            firstMap.put(first, secondMap);
        }
        Hashtable<String, Double> thirdMap = secondMap.get(second);
        if (thirdMap == null) {
            thirdMap = new Hashtable<String, Double>();
            secondMap.put(second, thirdMap);
        }
        Double d = thirdMap.get(third);
        if (d == null) {
            thirdMap.put(third, 1.0);
        }
        else {
            thirdMap.put(third, d + 1.0);
        }
        totalCount++;
    }

    public void subtract(double value) {
        for (Hashtable<String, Hashtable<String, Double>> secondMap : firstMap.values()) {
            for (Hashtable<String, Double> thirdMap : secondMap.values()) {
                for (Map.Entry<String, Double> entry : thirdMap.entrySet()) {
                    thirdMap.put(entry.getKey(), entry.getValue() - value);
                }
            }
        }
        totalCount -= totalCount * value;
    }

    public Double get(String first, String second, String third) {
        try {
            Double d = firstMap.get(first).get(second).get(third);
            if (d == null) {
                return 0.0;
            }
            return d;
        }
        catch (NullPointerException e) {
            return 0.0;
        }
    }

    public double getTotalCount() {
        return totalCount;
    }


    public static void main(String[] args) {
        FastTriCounter counter = new FastTriCounter();
        counter.increment("a", "b", "c");
        counter.increment("a", "b", "c");
        counter.increment("a", "b", "e");
        counter.subtract(0.5);
        System.out.println(counter.get("a", "b", "c"));
        System.out.println(counter.get("a", "b", "d"));
        System.out.println(counter.get("a", "b", "e"));
    }
}
