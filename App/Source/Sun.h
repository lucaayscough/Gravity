#pragma once


class Sun : public Planet{

//--------------------------------------------------//
// Constructors and destructors.

public:
    Sun(juce::OwnedArray<Planet>&, AudioContainer&, Parameters&);
    ~Sun() override;

//--------------------------------------------------//
// View methods.

    void paint(Graphics& g) override;
    void resized() override;
    void draw() override;

//--------------------------------------------------//
// Interface methods.

    void setPosXY(int, int) override;

    juce::ValueTree getState() override;
    int getDiameter() override;
    int getCentreX(Planet*) override;
    int getCentreY(Planet*) override;

//--------------------------------------------------//
// Controller methods.

private:
    bool hitTest(int, int) override;
    void mouseDown(const MouseEvent& e) override;
    void mouseUp(const MouseEvent& e) override;
    void mouseDrag(const MouseEvent& e) override;
    void mouseWheelMove(const MouseEvent&, const MouseWheelDetails&) override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Sun)
};
