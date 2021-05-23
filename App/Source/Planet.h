#pragma once

#include <JuceHeader.h>


class Planet: public juce::Component{
public:
    bool m_Destroy = false;

private:
    juce::ComponentDragger m_Dragger;

    const int M_SIZE_MODIFIER = 2;
    const int M_MAX_PLANET_SIZE = 100;
    const int M_MIN_PLANET_SIZE = 20;

    int m_Diameter = 50;
    int m_WindowWidth;
    int m_WindowHeight;
    juce::Value* m_DestroyPlanetPtr;

public:
    Planet();
    Planet(juce::Value* destroy_planet_ptr);
    Planet(const Planet&);
    ~Planet() override;

    void paint(Graphics& g) override;
    void resized() override;
    
    void reDraw(int diameter, int x, int y);
    void resizePlanet(int diameter);

    void setDiameter(int diameter);
    void setMapBoundaries(int width, int height);

    int getDiameter();

private:
    void mouseDown(const MouseEvent& e) override;
    void mouseDrag(const MouseEvent& e) override;
    void mouseWheelMove(const MouseEvent& e, const MouseWheelDetails& w) override;
    void visibilityChanged() override;
    
    void checkBounds();
};