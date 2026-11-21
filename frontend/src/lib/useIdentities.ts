import { useCallback, useEffect, useState } from "react";
import type { Identity } from "../api";
import { client } from "../client";

export interface IdentitiesState {
  items: Identity[];
  loading: boolean;
  error: string | null;
  reload: () => Promise<void>;
}

export function useIdentities(): IdentitiesState {
  const [items, setItems] = useState<Identity[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    setLoading(true);
    try {
      const list = await client.identities();
      setItems(list);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  return { items, loading, error, reload };
}
