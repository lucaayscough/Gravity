#pragma once


class Sun : public Planet{
public:
    // Constructors and destructors.
    Sun(juce::OwnedArray<Planet>&, AudioContainer&, Parameters&, ControlPanel&);
    void init() override;
    ~Sun() override;

public:
    // View methods.
    void paint(Graphics& g) override;
    void resized() override;
    void draw() override;

public:
    // Interface methods.
    void setPosXY(int, int) override;
    juce::ValueTree getState() override;

private:
    // Controller methods.
    bool hitTest(int, int) override;
    void mouseDown(const MouseEvent& e) override;
    void mouseUp(const MouseEvent& e) override;
    void mouseDrag(const MouseEvent& e) override;
    void mouseWheelMove(const MouseEvent&, const MouseWheelDetails&) override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Sun)
};
