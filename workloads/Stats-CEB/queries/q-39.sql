SELECT COUNT(*)
FROM comments AS c,
  postHistory AS ph,
  badges AS b,
  users AS u
WHERE u.Id = c.UserId
  AND u.Id = ph.UserId
  AND u.Id = b.UserId
  AND c.Score = 0
  AND c.CreationDate >= CAST('2010-07-20 10:52:57' AS timestamp)
  AND ph.PostHistoryTypeId = 5
  AND ph.CreationDate >= CAST('2011-01-31 15:35:37' AS timestamp)
  AND u.Reputation >= 1
  AND u.Reputation <= 356
  AND u.DownVotes <= 34
  AND u.CreationDate >= CAST('2010-07-19 21:29:29' AS timestamp)
  AND u.CreationDate <= CAST('2014-08-20 14:31:46' AS timestamp);
