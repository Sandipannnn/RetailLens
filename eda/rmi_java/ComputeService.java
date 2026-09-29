import java.rmi.Remote;
import java.rmi.RemoteException;

public interface ComputeService extends Remote {

    public int computeSquare(int number) throws RemoteException;

    public String fetchNodeTime() throws RemoteException;
}