package cs224n.util;


public class Triple<F, S, T> {
    F first;
    S second;
    T third;

    public F getFirst() {
        return first;
    }

    public S getSecond() {
        return second;
    }

    public T getThird() {
        return third;
    }

    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Triple)) return false;

        @SuppressWarnings("unchecked")
        final Triple triple = (Triple) o;

        if (first != null ? !first.equals(triple.first) : triple.first != null) return false;
        if (second != null ? !second.equals(triple.second) : triple.second != null) return false;
        if (third != null ? !third.equals(triple.third) : triple.third != null) return false;

        return true;
    }
}
