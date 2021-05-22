#pragma once

#include "Planet.h"


class Sun : public Planet{
private:
    const int M_DIAMETER = 75;

public:
    Sun();
    ~Sun() override;

    std::function<void()> getNewSample;
    void paint(Graphics& g) override;
    void resized() override;
    int getDiameter();

private:
    void mouseDown(const MouseEvent& e) override;
    void mouseDrag(const MouseEvent& e) override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Sun)
};