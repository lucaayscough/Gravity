#include "Headers.h"


//--------------------------------------------------//
// Constructors and destructors.

Planet::Planet(juce::OwnedArray<Planet>& planets_ref, AudioContainer& audiocontainer_ref, Parameters& parameters_ref, ControlPanel& controlpanel_ref)
    :   Astro(audiocontainer_ref, parameters_ref, controlpanel_ref),
        m_PlanetsRef(planets_ref){
    Logger::writeToLog("Planet created.");

    m_ColourGradient.addColour((double)0.0, juce::Colours::white);
    m_ColourGradient.addColour((double)0.2, juce::Colours::yellow);
    m_ColourGradient.addColour((double)0.4, juce::Colours::orange);
    m_ColourGradient.addColour((double)0.7, juce::Colours::red);
    m_ColourGradient.addColour((double)1.0, juce::Colours::darkred);
}

Planet::~Planet(){
    Logger::writeToLog("Planet destroyed.");
}

//--------------------------------------------------//
// View methods.

void Planet::paint(Graphics& g){
    if(getState().getProperty(Parameters::isActiveProp)){
        g.setColour(juce::Colours::green);
    }
    else{
        double max_distance = sqrt((double)(pow(getParentWidth() / 2, 2)) + (double)(pow(getParentHeight() / 2, 2)));
        double pos = (getDistance(getCentreX(), getCentreY(), getParentWidth() / 2, getParentHeight() / 2)) / max_distance;
        
        g.setColour(m_ColourGradient.getColourAtPosition(pos));
    }

    g.fillEllipse(
        getClipBoundary() / 2,
        getClipBoundary() / 2,
        getDiameter(),
        getDiameter()
    );
}

void Planet::resized(){
    draw(getDiameter(), getX(), getY());
    setPosXY(getX(), getY());
}

void Planet::draw(){setBounds(getPosX(), getPosY(), getDiameter() + getClipBoundary(), getDiameter() + getClipBoundary());}
void Planet::draw(int diameter, int x, int y){setBounds(x, y, diameter + getClipBoundary(), diameter + getClipBoundary());}

void Planet::resizePlanet(int diameter){
    int new_x;
    int new_y;

    if(diameter > getDiameter()){
        new_x = getX() - (Variables::SIZE_MODIFIER / 2);
        new_y = getY() - (Variables::SIZE_MODIFIER / 2);
    }
    else{
        new_x = getX() + (Variables::SIZE_MODIFIER / 2);
        new_y = getY() + (Variables::SIZE_MODIFIER / 2);
    }

    setDiameter(diameter);
    draw(diameter, new_x, new_y);

    // TODO:
    // NEED A WAY TO UPDATE GRAPH WHEN DONE ZOOMING IN ON SOUND
    //updateGraph();
}

void Planet::checkCollision(){
    int centrePosX = getX() + (getClipBoundary() + getDiameter()) / 2;
    int centrePosY = getY() + (getClipBoundary() + getDiameter()) / 2;

    float distance, minDistance;

    // Check collision with sun.
    {
        int centreXSun = getParentWidth() / 2;
        int centreYSun = getParentHeight() / 2;
        int sunDiameter = Variables::SUN_DIAMETER;

        distance = getDistance(centrePosX, centrePosY, centreXSun, centreYSun);
        minDistance = (sunDiameter + getDiameter()) / 2;

        if(distance <= minDistance){
            draw(getDiameter(), getPosX(), getPosY());
        }
    }

    Planet* planet;
    int centrePosX2, centrePosY2;

    for(int i = 0; i < m_PlanetsRef.size(); i++){
        // Variable for ease of use.
        planet = m_PlanetsRef[i];

        // Avoid self collision testing.
        if(planet->getComponentID() != getComponentID()){
            centrePosX2 = planet->getX() + (planet->getClipBoundary() + planet->getDiameter()) / 2;
            centrePosY2 = planet->getY() + (planet->getClipBoundary() + planet->getDiameter()) / 2;

            distance = getDistance(centrePosX, centrePosY, centrePosX2, centrePosY2);
            minDistance = (planet->getDiameter() + getDiameter()) / 2;

            if(distance <= minDistance){
                draw(getDiameter(), getPosX(), getPosY());
            }
        }
    }
}

void Planet::checkBounds(){
    //Check left boundary.
    if(getX() < -(getClipBoundary() / 2))
        draw(getDiameter(), -(getClipBoundary() / 2), getY());

    // Check top boundary.
    if(getY() < -(getClipBoundary() / 2))
        draw(getDiameter(), getX(), -(getClipBoundary() / 2));

    // Check right boundary,
    if(getX() + getDiameter() + (getClipBoundary() / 2) > getParentWidth())
        draw(getDiameter(), getParentWidth() - getDiameter() - (getClipBoundary() / 2), getY());

    // Check bottom boundary.
    if(getY() + getDiameter() + (getClipBoundary() / 2) > getParentHeight())
        draw(getDiameter(), getX(), getParentHeight() - getDiameter() - (getClipBoundary() / 2));
}

//--------------------------------------------------//
// Interface methods.

juce::ValueTree Planet::getState(){return m_ParametersRef.getRootPlanetNode().getChildWithProperty(Parameters::idProp, getComponentID());}
int Planet::getClipBoundary(){return Variables::CLIP_BOUNDARY;}

void Planet::setCentrePosXY(int x, int y){
    getState().setProperty(Parameters::posCentreXProp, x + getClipBoundary() / 2, nullptr);
    getState().setProperty(Parameters::posCentreYProp, y + getClipBoundary() / 2, nullptr);
}

//--------------------------------------------------//
// Controller methods.

bool Planet::hitTest(int x, int y){
    float a = pow((float)x - ((float)getDiameter() + (float)getClipBoundary()) / 2.0f, 2.0f);
    float b = pow((float)y - ((float)getDiameter() + (float)getClipBoundary()) / 2.0f, 2.0f);
    return sqrt(a + b) <= getDiameter() / 2;
}

void Planet::mouseDown(const MouseEvent& e){m_Dragger.startDraggingComponent(this, e);}

void Planet::mouseUp(const MouseEvent& e){
    if(e.mods.isLeftButtonDown()){
        // Generates new sample if double clicked with left mouse button.
        if(e.getNumberOfClicks() > 1){generateSample();}
        
        // Plays sample if clicked once with left mouse button.
        else if(e.getNumberOfClicks() == 1 && e.mouseWasClicked()){playSample();}

        // Updates latent mixture graph if there has been a dragging motion.
        else if(e.mouseWasDraggedSinceMouseDown()){updateGraph();}
    }
    
    // Destroys planet if clicked with right mouse button.
    else if(e.mods.isRightButtonDown()){
        m_ParametersRef.removePlanetNode(getComponentID());
    }
}

void Planet::mouseDrag(const MouseEvent& e){
    m_Dragger.dragComponent(this, e, nullptr);
    checkCollision();
    checkBounds();
    setPosXY(getX(), getY());
}

void Planet::mouseWheelMove(const MouseEvent& e, const MouseWheelDetails& w){
    juce::ignoreUnused(e);
    Logger::writeToLog("Wheel moved.");

    if(w.deltaY > 0.0f && getDiameter() < Variables::MAX_PLANET_SIZE){resizePlanet(getDiameter() + Variables::SIZE_MODIFIER);}
    else if(w.deltaY < 0.0f && getDiameter() > Variables::MIN_PLANET_SIZE){resizePlanet(getDiameter() - Variables::SIZE_MODIFIER);}
}

//--------------------------------------------------//
// Callback methods.

void Planet::visibilityChanged(){}

void Planet::valueChanged(juce::Value& value){
    juce::ignoreUnused(value);
    repaint();
}
