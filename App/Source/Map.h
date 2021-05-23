#pragma once

#include <JuceHeader.h>
#include "Planet.h"


class Map: public juce::Component, public juce::Value::Listener{
private:
    OwnedArray<Planet> m_Planets;
    const int M_MAX_NUM_PLANETS = 20;
    int m_NumPlanets = 0;
    juce::Value m_DestroyPlanet;

public:
    Map();
    ~Map() override;

    void paint(Graphics& g) override;
    void resized() override;

private:
    void createPlanet(int x, int y);
    void destroyPlanet();

    void reservePlanetMemory();

    void mouseDoubleClick(const MouseEvent& e) override;
    void valueChanged(juce::Value &value) override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Map)
};