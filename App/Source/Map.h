#pragma once


class Map: public juce::Component, public juce::Value::Listener{
private:

    // Planet container and variables.
    OwnedArray<Planet> m_Planets;
    int m_NumPlanets = 0;

    AudioContainer* m_AudioContainerPtr;
    Parameters* m_ParametersPtr;

    // Sun container.
    Sun m_Sun;

public:
    // Constructors and destructors.
    Map();
    Map(AudioContainer*, Parameters*);
    ~Map() override;

private:
    // View methods.
    void paint(Graphics&) override;
    void resized() override;
    
    void createSun();
    void createPlanet(int, int);
    void setPlanetID(Planet*);
    void setupPlanet(Planet*, int x, int y, juce::ValueTree);
    void destroyPlanet();

    // Interface methods
    int getMaxNumPlanets();
    float getDistance(Sun&, Planet*);
    float getDistance(Planet*, Planet*);
    float getForceVector(Sun&, Planet*);
    float getForceVector(Planet*, Planet*);

    // Mixes all latents.
    void mixLatents();

    // Controller methods.
    void mouseUp(const MouseEvent&) override;
    void mouseDoubleClick(const MouseEvent&) override;
    void valueChanged(juce::Value&) override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Map)
};