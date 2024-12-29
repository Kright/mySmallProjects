package test;

public class MutablePoint {
    public double x;
    public double y;
    public double z;
    public double w;

    public MutablePoint(double x, double y, double z, double w) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.w = w;
    }

    public MutablePoint copy() {
        return new MutablePoint(x, y, z, w);
    }

    public MutablePoint add(MutablePoint p) {
        x += p.x;
        y += p.y;
        z += p.z;
        w += p.w;
        return this;
    }

    public MutablePoint sub(MutablePoint p) {
        x -= p.x;
        y -= p.y;
        z -= p.z;
        w -= p.w;
        return this;
    }
}
