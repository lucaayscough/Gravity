#pragma once

#include <JuceHeader.h>


class Planet: public juce::Component{
private:
    juce::ComponentDragger m_dragger;
    int m_diameter;
    int m_window_width;
    int m_window_height;

public:
    Planet();
    ~Planet() override;

    void paint(Graphics& g) override;
    void resized() override;
    
    void setDiameter(int diameter);
    void setEdges(int width, int height);

private:
    void mouseDown(const MouseEvent& e);
    void mouseDrag(const MouseEvent& e);
    void checkBounds();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Planet)
};


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