#pragma once

#include <JuceHeader.h>


class Planet : public juce::Component{
    juce::ComponentDragger dragger;
    void paint(Graphics& g) override;
    void mouseDown(const MouseEvent& e);
    void mouseDrag(const MouseEvent& e);
};