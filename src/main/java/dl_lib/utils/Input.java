package dl_lib.utils;

public class Input {

    private int layerNum;

    private int nChannels = 1;

    public Input(int layerNum) {
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
