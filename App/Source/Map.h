#pragma once


class Map: public juce::Component, public juce::Value::Listener{
private:
    // Sun container.
    juce::OwnedArray<Sun> m_Sun;

    // Planet container and variables.
    OwnedArray<Planet> m_Planets;
    const int M_MAX_NUM_PLANETS = 20;
    int m_NumPlanets = 0;

    Generator* m_GeneratorPtr;
    AudioContainer* m_AudioContainerPtr;

public:
    Map();
    Map(Generator*, AudioContainer*);
    ~Map() override;
    
    void paint(Graphics&) override;
    void resized() override;
    void createSun();

private:
    void createPlanet(int, int);
    void setPlanetID(Planet*);
    void setupPlanet(Planet*, int x, int y);
    void destroyPlanet();

    // Returns distance between a planet and the sun.
    float getDistance(Sun*, Planet*);

    // Returns distance between a planet and the sun.
    float getDistance(Planet*, Planet*);

    // Returns force vector between a planet and the sun.
    float getForceVector(Sun*, Planet*);

    // Returns force vector between two planets.
    float getForceVector(Planet*, Planet*);

    // Mixes all latents.
    void mixLatents();

    void mouseUp(const MouseEvent&) override;
    void mouseDoubleClick(const MouseEvent&) override;
    void valueChanged(juce::Value&) override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Map)
};