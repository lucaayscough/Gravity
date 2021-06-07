#pragma once


class Planet: public juce::Component{
public:
    // Value used to activate planet destruction.
    juce::Value m_Destroy;  

    // Value used to activate lerp graph calculation.
    juce::Value m_LerpGraph;

    // Generated latents.
    at::Tensor m_LerpLatents;

private:
    juce::ComponentDragger m_Dragger;

protected:
    // Pointer to array containing planets.
    juce::OwnedArray<Planet>* m_PlanetsPtr;

    // Used to access the generator instantiated in
    // the PluginProcessor.
    AudioContainer* m_AudioContainerPtr;

    juce::ValueTree m_State;


//--------------------------------------------------//
// Constructors and destructors.

public:
    Planet();
    Planet(juce::OwnedArray<Planet>*, AudioContainer*, juce::ValueTree);
    ~Planet() override;

//--------------------------------------------------//
// View methods.

    void paint(Graphics&) override;
    void resized() override;
    virtual void draw();
    virtual void draw(int, int, int);

private:
    void resizePlanet(int);

//--------------------------------------------------//
// Interface methods.

private:
    void setDiameter(int);

public:
    virtual void setPosXY(int, int);
    virtual void setCentrePosXY(int, int);

public:
    virtual int getDiameter();
    virtual int getPosX();
    virtual int getPosY();

private:
    int getMapWidth();
    int getMapHeight();

public:
    int getClipBoundary();

private:
    float getDistance(int, int, int, int);
    float getDistance(Planet*, Planet*);

public:
    virtual int getCentreX(Planet*);
    virtual int getCentreY(Planet*);

private:
    void updateGraph();

public:
    virtual void addSample();
    virtual void playSample();

private:
    void checkCollision();
    void checkBounds();

//--------------------------------------------------//
// Controller methods.

private:
    bool hitTest(int, int) override;
    void mouseDown(const MouseEvent&) override;
    void mouseUp(const MouseEvent&) override;
    void mouseDrag(const MouseEvent&) override;
    void mouseWheelMove(const MouseEvent&, const MouseWheelDetails&) override;
    void visibilityChanged() override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Planet)
};
