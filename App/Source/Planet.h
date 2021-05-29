#pragma once


class Planet: public juce::Component{
public:
    // Value used to activate planet destruction.
    juce::Value m_Destroy;  

private:
    juce::ComponentDragger m_Dragger;

    const int M_SIZE_MODIFIER = 2;
    const int M_MAX_PLANET_SIZE = 100;
    const int M_MIN_PLANET_SIZE = 20;

    // Diameter of the planet.
    int m_Diameter = 50;

    // Boundary to avoid clipping of component when moved.
    int m_ClipBoundary = 100;

    // Map boundaries.
    int m_MapWidth;
    int m_MapHeight;

    // Collision safety.
    int m_PosX;
    int m_PosY;

protected:
    // Used to access the generator instantiated in
    // the PluginProcessor.
    Generator* m_GeneratorPtr;

    // Pointer to array containing planets.
    juce::OwnedArray<Planet>* m_PlanetsPtr;  

    // Generated sound.
    at::Tensor m_Latents;
    juce::Array<float> m_Sample;


public:
    Planet();
    Planet(juce::OwnedArray<Planet>*, Generator*);
    ~Planet() override;

    void paint(Graphics&) override;
    void resized() override;
    
    void draw(int, int, int);
    void resizePlanet(int);

    void setDiameter(int);
    void setMapBoundaries(int, int);
    void setPosXY(int, int);

    virtual int getDiameter();
    int getClipBoundary();
    float getDistance(int, int, int, int);

    virtual void generateLatents();
    virtual void generateSample();

private:
    bool hitTest(int, int) override;
    void mouseDown(const MouseEvent&) override;
    void mouseUp(const MouseEvent&) override;
    void mouseDrag(const MouseEvent&) override;
    void mouseWheelMove(const MouseEvent&, const MouseWheelDetails&) override;
    void visibilityChanged() override;

    void checkCollision();
    void checkBounds();
    void allocateStorage();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Planet)
};