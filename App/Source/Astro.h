#pragma once


class Astro: public juce::Component, juce::Value::Listener{
public: 
    // Constructors and destructors.
    Astro(AudioContainer&, Parameters&, ControlPanel&);
    ~Astro() override;
   
public:
    // View methods.
    virtual void draw() = 0;
    virtual void draw(int, int, int) = 0;

    // Interface methods.
    virtual void setDiameter(int);
    virtual void setPosXY(int, int);
    virtual void setCentrePosXY(int, int);

    virtual juce::ValueTree getState() = 0;
    virtual int getDiameter();
    virtual int getPosX();
    virtual int getPosY();
    virtual int getCentreX();
    virtual int getCentreY();

    virtual float getDistance(int, int, int, int);
    virtual float getDistance(Astro*, Astro*);

    virtual void updateGraph();
    virtual void generateSample();

    virtual void playSample();

protected:
    // Member variables.
    AudioContainer& m_AudioContainerRef;
    Parameters& m_ParametersRef;
    ControlPanel& m_ControlPanelRef;

    /*
    // Animator class.
    class Animator: juce::Timer{
    public:
        // Constructors and destructors.
        Animator();
        ~Animator() override;

        // Callback methods.
        void timerCallback() override;

        // Member variables.
        juce::Value m_AnimateDiameter;
        bool m_DiameterDirection = true;
    };

    // Member variables.
    Animator m_Animator;
    */

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Astro)
};


