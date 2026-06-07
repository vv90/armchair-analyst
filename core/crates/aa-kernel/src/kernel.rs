use aa_framework::prelude::*;

struct Kernel;
struct ProductionRuntime;

struct State {
    count: i32,
    active: bool,
}

#[derive(Debug)]
enum Event {
    CountChanged(i32),
}

#[derive(Debug)]
enum Effect {
    GetCount,
}

enum Subscription {
    TickSubscription,
}

impl Application for Kernel {
    type State = State;

    type Input = Event;

    type Effect = Effect;

    type Subscription = Subscription;

    fn init() -> Transition<Self::State, Self::Effect> {
        Transition {
            state: State {
                count: 0,
                active: false,
            },
            effects: vec![Effect::GetCount],
        }
    }

    fn transition(state: Self::State, input: Self::Input) -> Transition<Self::State, Self::Effect> {
        match input {
            Event::CountChanged(new_count) => Transition {
                state: State {
                    count: new_count,
                    ..state
                },
                effects: vec![],
            },
        }
    }

    fn subscriptions() -> Vec<Self::Subscription> {
        Vec::new()
    }
}

impl Runtime<Kernel> for ProductionRuntime {
    fn execute_effect(
        &self,
        effect: <Kernel as Application>::Effect,
    ) -> Vec<<Kernel as Application>::Input> {
        match effect {
            Effect::GetCount => {
                // Simulate fetching count from an external source
                vec![Event::CountChanged(42)]
            }
        }
    }

    fn spawn_subscription(
        &self,
        _sender: &std::sync::mpsc::Sender<<Kernel as Application>::Input>,
        _subscription: <Kernel as Application>::Subscription,
    ) {
    }

    fn log(&self, error: ApplicationError<<Kernel as Application>::Input>) {
        eprintln!("{:?}", error);
    }
}
