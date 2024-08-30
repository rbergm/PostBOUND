SELECT COUNT(*)
FROM postHistory AS ph, posts AS p, users AS u
WHERE p.OwnerUserId = u.Id
  AND ph.UserId = u.Id
  AND ph.CreationDate >= CAST('2010-08-21 05:30:40' AS timestamp)
  AND p.Score >= 0
  AND u.Reputation >= 1
  AND u.UpVotes <= 198
  AND u.CreationDate >= CAST('2010-07-19 20:49:05' AS timestamp);
