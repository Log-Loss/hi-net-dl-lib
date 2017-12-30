package dl_lib.utils;

public class Output {

    private int layerNum;

    private int nChannels = 1;

    public Output(int layerNum) {
        this.layerNum = layerNum;
    }

    public int getLayerNum() {
        return layerNum;
    }

    public void setnChannels(int nChannels) {
        this.nChannels = nChannels;
    }

    public int getnChannels() {
        return nChannels;
    }
}
