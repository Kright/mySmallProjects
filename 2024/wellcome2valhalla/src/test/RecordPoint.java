package test;

public record RecordPoint(double x, double y, double z, double w) {

    public RecordPoint add(RecordPoint p) {
        return new RecordPoint(x + p.x, y + p.y, z + p.z, w + p.w);
    }

    public RecordPoint sub(RecordPoint p) {
        return new RecordPoint(x - p.x, y - p.y, z - p.z, w - p.w);
    }
}
