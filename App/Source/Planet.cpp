#include "Headers.h"


//--------------------------------------------------//
// Constructors and destructors.

Planet::Planet(juce::OwnedArray<Planet>& planets_ref, AudioContainer& audiocontainer_ref, Parameters& parameters_ref)
    :   m_PlanetsRef(planets_ref),
        m_AudioContainerRef(audiocontainer_ref),
        m_ParametersRef(parameters_ref){
    Logger::writeToLog("Planet created.");
}

Planet::~Planet(){Logger::writeToLog("Planet destroyed.");}

//--------------------------------------------------//
// View methods.

void Planet::paint(Graphics& g){
    if(getState().getProperty(Parameters::isActiveProp)){
        g.setColour(juce::Colours::green);
        Logger::writeToLog("green");
    }
    else{
        g.setColour(juce::Colours::red);
        Logger::writeToLog("red");
    }
    
    draw(getDiameter(), getX(), getY());
    g.fillEllipse(getClipBoundary() / 2, getClipBoundary() / 2, getDiameter(), getDiameter());
    Logger::writeToLog("sdsd");
}

void Planet::resized(){}
void Planet::draw(){setBounds(getPosX(), getPosY(), getDiameter() + getClipBoundary(), getDiameter() + getClipBoundary());}
void Planet::draw(int diameter, int x, int y){setBounds(x, y, diameter + getClipBoundary(), diameter + getClipBoundary());}

void Planet::resizePlanet(int diameter){
    int new_x;
    int new_y;

    if(diameter > getDiameter()){
        new_x = getX() - (Variables::SIZE_MODIFIER / 2);
        new_y = getY() - (Variables::SIZE_MODIFIER / 2);
    } else{
        new_x = getX() + (Variables::SIZE_MODIFIER / 2);
        new_y = getY() + (Variables::SIZE_MODIFIER / 2);
    }

    setDiameter(diameter);
    draw(diameter, new_x, new_y);

    // TODO:
    // NEED A WAY TO UPDATE GRAPH WHEN DONE ZOOMING IN ON SOUND
    //updateGraph();
}

void Planet::setDiameter(int diameter){getState().setProperty(Parameters::diameterProp, diameter, nullptr);}
void Planet::setMapSize(int width, int height){
    getState().setProperty(Parameters::mapWidthProp, width, nullptr);
    getState().setProperty(Parameters::mapHeightProp, height, nullptr);
}

void Planet::setPosXY(int x, int y){
    getState().setProperty(Parameters::posXProp, x, nullptr);
    getState().setProperty(Parameters::posYProp, y, nullptr);
}

void Planet::setCentrePosXY(int x, int y){
    getState().setProperty(Parameters::posCentreXProp, x, nullptr);
    getState().setProperty(Parameters::posCentreYProp, y, nullptr);
}

juce::ValueTree Planet::getState(){return m_ParametersRef.getRootPlanetNode().getChildWithProperty(Parameters::idProp, getComponentID());}
int Planet::getDiameter(){return getState().getProperty(Parameters::diameterProp);}
int Planet::getPosX(){return getState().getProperty(Parameters::posXProp);}
int Planet::getPosY(){return getState().getProperty(Parameters::posYProp);}
int Planet::getMapWidth(){return getState().getProperty(Parameters::mapWidthProp);}
int Planet::getMapHeight(){return getState().getProperty(Parameters::mapHeightProp);}
int Planet::getClipBoundary(){return Variables::CLIP_BOUNDARY;}

float Planet::getDistance(int xa, int ya, int xb, int yb){  
    float a = (float)pow(xb - xa, 2);
    float b = (float)pow(yb - ya, 2); 
    return sqrt(a + b);
}

float Planet::getDistance(Planet* planet_a, Planet* planet_b){  
    int centreXA = getCentreX(planet_a);
    int centreYA = getCentreY(planet_a);
    int centreXB = getCentreX(planet_b);
    int centreYB = getCentreY(planet_b);

    float a = (float)pow(centreXB - centreXA, 2);
    float b = (float)pow(centreYB - centreYA, 2);

    return sqrt(a + b);
}

int Planet::getCentreX(Planet* planet){return planet->getX() + ((planet->getDiameter() + planet->getClipBoundary()) / 2);}
int Planet::getCentreY(Planet* planet){return planet->getY() + ((planet->getDiameter() + planet->getClipBoundary()) / 2);}
void Planet::updateGraph(){getState().setProperty(Parameters::updateGraphSignal, true, nullptr);}
void Planet::generateSample(){getState().setProperty(Parameters::generateSampleSignal, true, nullptr);}

void Planet::playSample(){
    Logger::writeToLog("Playing audio...");
    m_ParametersRef.setActivePlanet(getState());
    m_AudioContainerRef.sampleIndex.clear();
    m_AudioContainerRef.playAudio = true;
}

//--------------------------------------------------//
// Private methods.

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
    setCentrePosXY(getCentreX(this), getCentreY(this));
}

void Planet::mouseWheelMove(const MouseEvent& e, const MouseWheelDetails& w){
    juce::ignoreUnused(e);
    Logger::writeToLog("Wheel moved.");

    if(w.deltaY > 0.0f && getDiameter() < Variables::MAX_PLANET_SIZE){resizePlanet(getDiameter() + Variables::SIZE_MODIFIER);}
    else if(w.deltaY < 0.0f && getDiameter() > Variables::MIN_PLANET_SIZE){resizePlanet(getDiameter() - Variables::SIZE_MODIFIER);}
}

void Planet::visibilityChanged(){}

void Planet::checkCollision(){
    int centrePosX = getCentreX(this);
    int centrePosY = getCentreY(this);

    float distance, minDistance;

    // Check collision with sun.
    {
        int centreXSun = Variables::WINDOW_WIDTH / 2;
        int centreYSun = Variables::WINDOW_HEIGHT / 2;
        int sunDiameter = Variables::SUN_DIAMETER;

        distance = getDistance(centrePosX, centrePosY, centreXSun, centreYSun);
        minDistance = (sunDiameter + getDiameter()) / 2;

        if(distance <= minDistance){draw(getDiameter(), getPosX(), getPosY());}
    }

    Planet* planet;
    int centrePosX2, centrePosY2;

    for(int i = 0; i < m_PlanetsRef.size(); i++){
        // Variable for ease of use.
        planet = m_PlanetsRef[i];

        // Avoid self collision testing.
        if(planet->getComponentID() != getComponentID()){
            centrePosX2 = getCentreX(planet);
            centrePosY2 = getCentreY(planet);

            distance = getDistance(centrePosX, centrePosY, centrePosX2, centrePosY2);
            minDistance = (planet->getDiameter() + getDiameter()) / 2;

            if(distance <= minDistance){draw(getDiameter(), getPosX(), getPosY());}
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
    if(getX() + getDiameter() + (getClipBoundary() / 2) > getMapWidth())
        draw(getDiameter(), getMapWidth() - getDiameter() - (getClipBoundary() / 2), getY());

    // Check bottom boundary.
    if(getY() + getDiameter() + (getClipBoundary() / 2) > getMapHeight())
        draw(getDiameter(), getX(), getMapHeight() - getDiameter() - (getClipBoundary() / 2));
}
