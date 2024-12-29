package test;

public value class ValuePoint {
    public final double x;
    public final double y;
    public final double z;
    public final double w;

    public ValuePoint(double x, double y, double z, double w) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.w = w;
    }

    public ValuePoint add(ValuePoint p) {
        return new ValuePoint(x + p.x, y + p.y, z + p.z, w + p.w);
    }

    public ValuePoint sub(ValuePoint p) {
        return new ValuePoint(x - p.x, y - p.y, z - p.z, w - p.w);
    }
}
