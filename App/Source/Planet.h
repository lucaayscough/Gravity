#pragma once


class Planet: public juce::Component{
//--------------------------------------------------//
// Constructors and destructors.

public:
    Planet(juce::OwnedArray<Planet>&, AudioContainer&, Parameters&, ControlPanel&);
    virtual void init();
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

public:
    void setDiameter(int);
    void setMapSize(int, int);
    virtual void setPosXY(int, int);
    virtual void setCentrePosXY(int, int);

    virtual juce::ValueTree getState();
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
    virtual int getCentreX();
    virtual int getCentreY();
    virtual void updateGraph();
    virtual void generateSample();

public:
    virtual void playSample();

private:
    void checkCollision();
    void checkBounds();

//--------------------------------------------------//
// Controller methods.

private:
    bool hitTest(int, int) override;
    void mouseEnter(const MouseEvent&) override;
    void mouseExit(const MouseEvent&) override;
    void mouseDown(const MouseEvent&) override;
    void mouseUp(const MouseEvent&) override;
    void mouseDrag(const MouseEvent&) override;
    void mouseWheelMove(const MouseEvent&, const MouseWheelDetails&) override;
    void visibilityChanged() override;

private:
    // Member variables.
    juce::ComponentDragger m_Dragger;

protected:
    // Member variables.
    juce::OwnedArray<Planet>& m_PlanetsRef;
    AudioContainer& m_AudioContainerRef;
    Parameters& m_ParametersRef;
    ControlPanel& m_ControlPanelRef;
    juce::GlowEffect m_GlowEffect;
    juce::ColourGradient m_ColourGradient;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Planet)
};
