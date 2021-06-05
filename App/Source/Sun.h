#pragma once


class Sun : public Planet{
public:
    // Constructors and destructors.
    Sun();
    Sun(juce::OwnedArray<Planet>*, AudioContainer*, juce::ValueTree);
    ~Sun() override;

    // View methods.
    void paint(Graphics& g) override;
    void resized() override;
    void draw();

    // Interface methods.
    int getDiameter() override;

    // Temporary methods.
    void generateLatents() override;
    void generateSample(at::Tensor&) override;

private:
    // Controller methods.
    bool hitTest(int, int) override;
    void mouseDown(const MouseEvent& e) override;
    void mouseUp(const MouseEvent& e) override;
    void mouseDrag(const MouseEvent& e) override;
    void mouseWheelMove(const MouseEvent&, const MouseWheelDetails&) override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Sun)
};
