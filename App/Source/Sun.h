#pragma once


class Sun : public Planet{
private:
    const int M_DIAMETER = Variables::SUN_DIAMETER;

public:
    Sun();
    Sun(juce::OwnedArray<Planet>*, Generator*, AudioContainer*);
    ~Sun() override;

    void paint(Graphics& g) override;
    void resized() override;

    // When called the component is redrawn.
    void draw();

    int getDiameter() override;
    float getDistance(Planet*);
    
    void generateLatents() override;
    void generateSample(at::Tensor&) override;

private:
    bool hitTest(int, int) override;
    void mouseDown(const MouseEvent& e) override;
    void mouseUp(const MouseEvent& e) override;
    void mouseDrag(const MouseEvent& e) override;
    void mouseWheelMove(const MouseEvent&, const MouseWheelDetails&) override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Sun)
};
