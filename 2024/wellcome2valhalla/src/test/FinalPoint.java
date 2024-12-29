package test;

public class FinalPoint {
    public final double x, y, z, w;

    public FinalPoint(double x, double y, double z, double w) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.w = w;
    }

    public FinalPoint add(FinalPoint p) {
        return new FinalPoint(x + p.x, y + p.y, z + p.z, w + p.w);
    }

    public FinalPoint sub(FinalPoint p) {
        return new FinalPoint(x - p.x, y - p.y, z - p.z, w - p.w);
    }
}
