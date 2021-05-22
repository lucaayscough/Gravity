#pragma once

#include <JuceHeader.h>


class Planet: public juce::Component{
private:
    juce::ComponentDragger m_Dragger;
    int m_Diameter;
    int m_WindowWidth;
    int m_WindowHeight;

public:
    Planet();
    Planet(const Planet&);
    ~Planet() override;

    void paint(Graphics& g) override;
    void resized() override;
    
    void setDiameter(int diameter);
    void setWindowBoundary(int width, int height);

private:
    void mouseDown(const MouseEvent& e);
    void mouseDrag(const MouseEvent& e);
    void checkBounds();
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