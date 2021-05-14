#pragma once

#include <JuceHeader.h>


class Planet : public juce::Component{
    juce::ComponentDragger dragger;
    int diameter = 50;
    void paint(Graphics& g) override;
    void resized() override;
    void mouseDown(const MouseEvent& e);
    void mouseDrag(const MouseEvent& e);
};


class Sun : public Planet{
    public: const int DIAMETER = 75;
    public: std::function<void()> getNewSample;

    void paint(Graphics& g) override;
    void resized() override;
    void mouseDown(const MouseEvent& e) override;
    void mouseDrag(const MouseEvent& e) override;
};